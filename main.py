import random
import asyncio
import re
from html import unescape
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlencode, urljoin, urlparse, unquote

import astrbot.api.star as star
from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageChain

# Google GenAI SDK
from google import genai
from google.genai import types

# HTTP & HTML parsing
try:
	import httpx
except Exception:  # pragma: no cover - 延迟导入失败时给出友好提示
	httpx = None

try:
	from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
	BeautifulSoup = None

# 全局内存 LRU 缓存，避免对相同的 Google 接地重定向链接重复发起网络探测
_URL_CACHE: Dict[str, str] = {}
_TITLE_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 1000
_TITLE_HTML_LIMIT = 65536
_BROWSER_UA = (
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
	"(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def _cache_put(cache: Dict[str, str], key: str, value: str) -> None:
	"""将解析结果存入全局 LRU 缓存"""
	if len(cache) >= _MAX_CACHE_SIZE:
		try:
			cache.pop(next(iter(cache)))
		except Exception:
			cache.clear()
	cache[key] = value


def _is_domain_like(title: str) -> bool:
	t = (title or "").strip().lower().removeprefix("www.")
	if not t:
		return True
	if " " in t:
		return False
	return bool(re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", t))


def _clean_page_title(title: str) -> str:
	title = unescape(title or "")
	title = re.sub(r"\s+", " ", title).strip(" \t\r\n-|•·")
	parts = re.split(r"\s+[\|\-–—]\s+", title)
	if len(parts) >= 2 and len(parts[0]) >= 12:
		title = parts[0].strip()
	if len(title) > 80:
		title = title[:80].rstrip() + "…"
	return title


def _title_from_url_slug(url: str) -> str:
	try:
		path = unquote(urlparse(url).path or "")
	except Exception:
		return ""
	parts = [p for p in path.split("/") if p]
	skip = {
		"index", "news", "article", "articles", "story", "post", "posts",
		"blog", "en", "en-global", "zh", "pr", "press", "html", "home",
		"content", "detail", "details", "page",
	}
	cand = ""
	for part in reversed(parts):
		part = re.sub(r"\.(html?|php|aspx?)$", "", part, flags=re.I)
		low = part.lower()
		if low in skip or re.fullmatch(r"\d{4,}", part):
			continue
		# 跳过日期/纯编号：08-01、2026-09-08、12345
		if re.fullmatch(r"\d{1,4}([-_]\d{1,2}){1,3}", low):
			continue
		letters = re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", part)
		if len(letters) < 4:
			continue
		if len(part) >= 8 or "-" in part or "_" in part or re.search(r"[\u4e00-\u9fff]", part):
			cand = part
			break
	if not cand:
		return ""
	s = re.sub(r"[\-_]+", " ", cand)
	s = re.sub(r"\s+", " ", s).strip()
	if re.fullmatch(r"[A-Za-z0-9 .,'’:+]+", s):
		s = s.title()
	return s[:80]


def _extract_html_title(html: str) -> str:
	if BeautifulSoup is not None:
		soup = BeautifulSoup(html, "html.parser")
		og = soup.find("meta", attrs={"property": "og:title"}) or soup.find(
			"meta", attrs={"name": "og:title"}
		)
		if og and og.get("content"):
			return str(og.get("content")).strip()
		tw = soup.find("meta", attrs={"name": "twitter:title"})
		if tw and tw.get("content"):
			return str(tw.get("content")).strip()
		if soup.title and soup.title.string:
			return str(soup.title.string).strip()
		h1 = soup.find("h1")
		if h1:
			t = h1.get_text(" ", strip=True)
			if t:
				return t
	og = re.search(
		r'<meta\b[^>]*?(?:property|name)=["\']og:title["\'][^>]*?content=["\']([^"\']+)["\']',
		html,
		re.I | re.S,
	) or re.search(
		r'<meta\b[^>]*?content=["\']([^"\']+)["\'][^>]*?(?:property|name)=["\']og:title["\']',
		html,
		re.I | re.S,
	)
	if og:
		return og.group(1).strip()
	tw = re.search(
		r'<meta\b[^>]*?name=["\']twitter:title["\'][^>]*?content=["\']([^"\']+)["\']',
		html,
		re.I | re.S,
	)
	if tw:
		return tw.group(1).strip()
	m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
	if m:
		return re.sub(r"<[^>]+>", "", m.group(1)).strip()
	return ""


class Main(star.Star):
	"""
	使用 Gemini 2.0 Flash + Google Search 原生工具进行联网检索。
	在函数工具中执行检索与汇总，返回结果文本，框架会把该文本作为 tool 消息注入当前对话，
	供主 LLM 后续综合使用。
	"""

	def __init__(self, context: star.Context, config=None) -> None:
		self.context = context
		# AstrBot 会根据 _conf_schema.json 构造 config（AstrBotConfig），此处按 dict 访问
		self.config = config or {}
		self._rr_index = 0  # 轮询下标
		self._clients: dict[str, genai.Client] = {}

	async def initialize(self):
		# 默认启用工具
		self.context.activate_llm_tool("gemini_search")
		self.context.activate_llm_tool("web_fetch")
		self.context.activate_llm_tool("webshot_analyze")
		self.context.activate_llm_tool("webshot_send")
		logger.info("[gemini_search] 函数工具已启用")
		logger.info("[web_fetch] 函数工具已启用")
		logger.info("[webshot_analyze] 函数工具已启用")
		logger.info("[webshot_send] 函数工具已启用")

	@llm_tool("gemini_search")
	async def gemini_search(self, event: AstrMessageEvent, query: str) -> str:
		"""这是一个“联网搜索”的函数工具（工具名：gemini_search）。当需要获取互联网上的实时/最新信息时，你必须调用本工具进行搜索。

		Args:
			query(string): 简要说明用户希望检索的查询内容

		Returns:
			str: 要点摘要与引用来源，作为 tool 消息注入上下文
		"""
		try:
			client = self._get_client()
		except Exception as e:
			logger.error(f"[gemini_search] 初始化客户端失败: {e}")
			return "请先在插件配置中填写有效的 Gemini API Key。"

		model = self.config.get("model", "gemini-2.0-flash")

		# 启用原生 Google Search 工具
		config = types.GenerateContentConfig(
			tools=[types.Tool(google_search=types.GoogleSearch())],
			temperature=0.2,
		)

		# 净化提示词：专注生成客观要点摘要，不再要求模型输出参考来源或 URL
		# 来源将由插件从官方 API 响应结构 candidates[0].grounding_metadata 中结构化提取与还原
		prompt = (
			"你是检索聚合助手。请使用 Google Search 工具对下述问题进行检索，"
			"产出包含关键要点的条目式摘要。\n"
			"请避免冗长描述，直接给出客观准确的结论。\n"
			"问题：" + query
		)

		try:
			resp = await client.models.generate_content(
				model=model,
				contents=prompt,
				config=config,
			)
			text = getattr(resp, "text", None) or self._extract_text(resp)
			if not text:
				return "未从检索中获得可用文本结果。"
			text = text.strip()

			grounding_meta = None
			candidates = getattr(resp, "candidates", None) or []
			if candidates:
				grounding_meta = getattr(candidates[0], "grounding_metadata", None)

			if grounding_meta:
				raw_sources: List[Tuple[str, str]] = []
				chunks = getattr(grounding_meta, "grounding_chunks", None) or []
				for chunk in chunks:
					web = getattr(chunk, "web", None)
					if not web:
						continue
					uri = getattr(web, "uri", None)
					if not uri:
						continue
					title = getattr(web, "title", None) or getattr(web, "domain", None) or "来源"
					raw_sources.append((str(title).strip(), uri))

				if raw_sources:
					should_resolve = bool(self.config.get("resolve_grounding_urls", True))
					if should_resolve and httpx is not None:
						resolved_sources = await self._resolve_sources(raw_sources)
					else:
						resolved_sources = raw_sources

					deduped_sources: List[Tuple[str, str]] = []
					seen_urls: set[str] = set()
					for title, link in resolved_sources:
						if link in seen_urls:
							continue
						seen_urls.add(link)
						deduped_sources.append((title, link))

					if deduped_sources:
						deduped_sources = await self._enrich_source_titles(deduped_sources)
						source_lines = ["", "302重定向链接如下"]
						for title, link in deduped_sources:
							source_lines.append(f"[{title}] {link}")
						text += "\n".join(source_lines)

				if bool(self.config.get("show_search_queries", False)):
					queries = getattr(grounding_meta, "web_search_queries", None) or []
					if queries:
						text += f"\n检索关键词: {', '.join(queries)}"

			return text
		except Exception as e:
			logger.error(f"[gemini_search] 调用失败: {e}", exc_info=True)
			return f"检索失败：{e}"

	async def _resolve_sources(self, raw_sources: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
		"""并发解析多个接地重定向链接"""
		if not raw_sources:
			return []

		proxy = (self.config.get("grounding_proxy") or "").strip() or None
		client_kwargs = {
			"headers": {
				"User-Agent": self.config.get("fetch_user_agent", "Mozilla/5.0 AstrBot")
			},
			"trust_env": True,
		}
		if proxy:
			client_kwargs["proxy"] = proxy

		try:
			async with httpx.AsyncClient(**client_kwargs) as client:
				tasks = [self._resolve_single_redirect(client, uri) for _, uri in raw_sources]
				resolved_urls = await asyncio.gather(*tasks, return_exceptions=True)

			results: List[Tuple[str, str]] = []
			for (title, raw_uri), res in zip(raw_sources, resolved_urls):
				if isinstance(res, Exception) or not res:
					results.append((title, raw_uri))
				else:
					results.append((title, str(res)))
			return results
		except Exception as e:
			logger.warning(f"[gemini_search] 并发解析重定向链接发生异常: {e}，回退使用原始链接")
			return raw_sources

	async def _resolve_single_redirect(self, client: httpx.AsyncClient, raw_url: str) -> str:
		"""解析单个 vertexaisearch 链接的目标真实 URL"""
		if not raw_url.startswith("https://vertexaisearch.cloud.google.com/grounding-api-redirect/"):
			return raw_url

		if raw_url in _URL_CACHE:
			return _URL_CACHE[raw_url]

		try:
			# 不跟随重定向，直接捕获 302/301/303/307/308 的 Location 响应头，单请求超时约 3.5 秒
			resp = await client.get(raw_url, follow_redirects=False, timeout=3.5)
			if resp.status_code in (301, 302, 303, 307, 308):
				location = resp.headers.get("Location")
				if location:
					resolved = urljoin(raw_url, location)
					_cache_put(_URL_CACHE, raw_url, resolved)
					return resolved
		except Exception as e:
			logger.debug(f"[gemini_search] 解析重定向链接失败 ({raw_url[:60]}...): {e}")

		# 失败优雅降级，退回原始 redirect URL
		return raw_url

	def _httpx_client_kwargs(self, extra_headers: Optional[dict] = None) -> dict:
		proxy = (self.config.get("grounding_proxy") or "").strip() or None
		headers = {
			"User-Agent": self.config.get("fetch_user_agent", "Mozilla/5.0 AstrBot")
		}
		if extra_headers:
			headers.update(extra_headers)
		kwargs = {"headers": headers, "trust_env": True}
		if proxy:
			kwargs["proxy"] = proxy
		return kwargs

	async def _enrich_source_titles(
		self, sources: List[Tuple[str, str]]
	) -> List[Tuple[str, str]]:
		"""把域名占位标题换成文章标题；抓取失败时退回 URL 路径名。"""
		if not sources or httpx is None:
			return sources

		need_fetch: List[int] = []
		prepared: List[Tuple[str, str]] = []
		for idx, (title, url) in enumerate(sources):
			cached = _TITLE_CACHE.get(url)
			if cached:
				prepared.append((cached, url))
				continue
			slug = _title_from_url_slug(url)
			fallback = slug or title
			if (
				url.startswith("https://vertexaisearch.cloud.google.com/")
				or not url.startswith(("http://", "https://"))
				or not _is_domain_like(title)
			):
				final_title = title if not _is_domain_like(title) else fallback
				prepared.append((final_title, url))
				if not _is_domain_like(final_title):
					_cache_put(_TITLE_CACHE, url, final_title)
				continue
			prepared.append((fallback, url))
			need_fetch.append(idx)

		if not need_fetch:
			return prepared

		try:
			async with httpx.AsyncClient(
				**self._httpx_client_kwargs(
					{
						"User-Agent": _BROWSER_UA,
						"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
						"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
					}
				)
			) as client:
				tasks = [
					self._fetch_page_title(client, prepared[i][1], prepared[i][0])
					for i in need_fetch
				]
				titles = await asyncio.gather(*tasks, return_exceptions=True)
			for i, res in zip(need_fetch, titles):
				fallback = prepared[i][0]
				url = prepared[i][1]
				if isinstance(res, Exception) or not res:
					title = fallback
				else:
					title = str(res)
				prepared[i] = (title, url)
				if title:
					_cache_put(_TITLE_CACHE, url, title)
		except Exception as e:
			logger.debug(f"[gemini_search] 抓取文章标题失败: {e}")
		return prepared

	async def _fetch_page_title(
		self, client: "httpx.AsyncClient", url: str, fallback: str
	) -> str:
		try:
			async with client.stream(
				"GET", url, follow_redirects=True, timeout=4.0
			) as resp:
				if resp.status_code >= 400:
					return fallback
				ctype = (resp.headers.get("content-type") or "").lower()
				if ctype and ("html" not in ctype) and ("xml" not in ctype) and ("text/" not in ctype):
					return fallback
				buf = bytearray()
				async for chunk in resp.aiter_bytes():
					buf.extend(chunk)
					if len(buf) >= _TITLE_HTML_LIMIT:
						break
			charset = "utf-8"
			m = re.search(r"charset=([\w-]+)", ctype or "", re.I)
			if m:
				charset = m.group(1)
			html = bytes(buf).decode(charset, errors="replace")
			title = _clean_page_title(_extract_html_title(html))
			if title and not _is_domain_like(title):
				return title
		except Exception as e:
			logger.debug(f"[gemini_search] 读取页面标题失败 ({url[:80]}): {e}")
		return fallback

	@llm_tool("web_fetch")
	async def web_fetch(self, event: AstrMessageEvent, url: str) -> str:
		"""抓取网页文本内容（去标签纯文本）。

		Args:
			url(string): 需要抓取内容的网页 URL

		Returns:
			str: 提取的纯文本（前 20,000 字符内），失败时返回错误信息
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"
		try:
			text = await self._fetch_page_text(url)
			if not text:
				return "未能从页面中提取到有效文本。"
			max_chars = int(self.config.get("fetch_max_chars", 20000))
			return text[:max_chars]
		except Exception as e:
			logger.error(f"[web_fetch] 抓取失败: {e}")
			return f"抓取失败：{e}"

	@llm_tool("webshot_analyze")
	async def webshot_analyze(self, event: AstrMessageEvent, url: str, prompt: str = "请根据网页截图进行要点提炼与关键信息抽取，输出条目式结论与可操作建议。") -> str:
		"""此工具用于对网页进行截图，根据配置选择返回图片给AI模型分析。

		Args:
			url(string): 网页 URL
			prompt(string): 对截图的分析提示词（可选）

		Returns:
			str: 模型对截图的分析文本，或图片标记（由主模型分析）
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"

		fmt = str(self.config.get("screenshot_format", "webp"))
		width = int(self.config.get("screenshot_width", 1920))
		height = int(self.config.get("screenshot_height", 1080))
		shot_urls = self._build_screenshot_urls(url, fmt=fmt, width=width, height=height)

		try:
			image_bytes, mime = await self._fetch_screenshot(shot_urls, fmt)
		except Exception as e:
			logger.error(f"[webshot_analyze] 截图获取失败: {e}")
			return f"截图失败：{e}"

		# 检查是否使用插件内Gemini分析
		use_gemini = bool(self.config.get("webshot_analyze_with_gemini", True))
		
		if not use_gemini:
			# 直接将图片注入到主模型的请求上下文中
			logger.info(f"[webshot_analyze] 将截图注入到主模型请求上下文中进行分析")
			try:
				import base64
				base64_str = base64.b64encode(image_bytes).decode('utf-8')
				# 获取当前的 ProviderRequest 并添加图片
				req = event.get_extra("provider_request")
				if req is not None:
					# 使用 base64:// 前缀，框架会自动处理
					req.image_urls.append(f"base64://{base64_str}")
					logger.info(f"[webshot_analyze] 成功将截图注入到 ProviderRequest.image_urls 中")
					return f"已获取网页 {url} 的截图并注入到上下文中。请根据截图内容回答用户问题。{prompt}"
				else:
					# 如果获取不到 ProviderRequest，回退到发送给用户
					logger.warning("[webshot_analyze] 无法获取 ProviderRequest，回退到发送图片给用户")
					await event.send(MessageChain().base64_image(base64_str))
					return f"已获取网页 {url} 的截图（已发送给用户）。{prompt}"
			except Exception as e:
				logger.error(f"[webshot_analyze] 注入截图失败: {e}")
				return f"注入截图失败：{e}"
		
		# 使用插件内Gemini分析
		try:
			client = self._get_client()
		except Exception as e:
			logger.error(f"[webshot_analyze] 初始化客户端失败: {e}")
			return "请先在插件配置中填写有效的 Gemini API Key。"

		try:
			image_part = self._make_image_part(image_bytes, mime)
			contents = [
				types.Content(
					role="user",
					parts=[image_part, types.Part.from_text(text=prompt)],
				)
			]
			model = self.config.get("model", "gemini-2.0-flash")
			resp = await client.models.generate_content(
				model=model,
				contents=contents,
			)
			text = getattr(resp, "text", None) or self._extract_text(resp)
			return text.strip() if text else "未从分析中获得可用文本结果。"
		except Exception as e:
			logger.error(f"[webshot_analyze] 调用失败: {e}")
			return f"分析失败：{e}"

	@llm_tool("webshot_send")
	async def webshot_send(self, event: AstrMessageEvent, url: str) -> str:
		"""对网页进行截图，并直接发送截图给用户。

		Args:
			url(string): 网页 URL
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"

		fmt = str(self.config.get("screenshot_format", "webp"))
		width = int(self.config.get("screenshot_width", 1920))
		height = int(self.config.get("screenshot_height", 1080))
		shot_urls = self._build_screenshot_urls(url, fmt=fmt, width=width, height=height)

		should_moderate = bool(self.config.get("moderation_before_image_send", False))
		if should_moderate:
			try:
				client = self._get_client()
			except Exception as e:
				logger.error(f"[webshot_send] 初始化客户端失败: {e}")
				return "请先在插件配置中填写有效的 Gemini API Key。"
			try:
				image_bytes, mime = await self._fetch_screenshot(shot_urls, fmt)
				image_part = self._make_image_part(image_bytes, mime)
				check_prompt = (
					"你是合规审核器。请仅输出一个词：ALLOW 或 BLOCK。"
					"当图片包含露骨色情、严重暴力、仇恨、隐私泄露、恶意程序二维码/链接等不适宜内容时输出 BLOCK；"
					"其余情况输出 ALLOW。"
				)
				contents = [
					types.Content(role="user", parts=[image_part, types.Part.from_text(text=check_prompt)])
				]
				model = self.config.get("model", "gemini-2.0-flash")
				resp = await client.models.generate_content(model=model, contents=contents)
				decision = (getattr(resp, "text", "") or self._extract_text(resp) or "").strip().upper()
				if "BLOCK" in decision and "ALLOW" not in decision:
					await event.send(MessageChain().message("由于合规审核未通过，图片已被拦截。"))
					return "图片已拦截（审核结果：BLOCK）。"
			except Exception as e:
				logger.error(f"[webshot_send] 审核失败: {e}")
				return f"审核失败：{e}"

		# 发送网络图片 URL（平台适配层会下载或转发）
		# 注意：这里发送的是第一个服务的URL，实际下载会自动轮换
		try:
			await event.send(MessageChain().url_image(shot_urls[0]))
			return f"截图已发送（使用 {len(shot_urls)} 个备用服务）"
		except Exception as e:
			logger.error(f"[webshot_send] 发送失败: {e}")
			return f"发送失败：{e}"

	def _get_client(self):
		"""根据配置选择 API Key，并创建/复用异步 client。支持随机或轮询策略。"""
		keys = self.config.get("api_key", []) or []
		if not keys:
			raise RuntimeError("请在插件配置中填写至少一个 Google API Key。")

		use_random = bool(self.config.get("random_api_key_selection", False))
		if use_random:
			key = random.choice(keys)
		else:
			key = keys[self._rr_index % len(keys)]
			self._rr_index += 1

		if key in self._clients:
			return self._clients[key]

		api_base = self.config.get(
			"api_base_url", "https://generativelanguage.googleapis.com"
		)
		if api_base.endswith("/"):
			api_base = api_base[:-1]

		http_options = types.HttpOptions(
			base_url=api_base,
		)
		client = genai.Client(api_key=key, http_options=http_options).aio
		self._clients[key] = client
		return client

	def _build_screenshot_urls(self, page_url: str, fmt: str = "webp", width: int = 1920, height: int = 1080) -> list[str]:
		"""构建所有配置的截图服务 URL 列表。"""
		bases = self.config.get(
			"screenshot_api_base", 
			["https://screenshotsnap.com/api/screenshot"]
		)
		# 兼容旧配置：如果是字符串则转为列表
		if isinstance(bases, str):
			bases = [bases]
		
		params = {
			"url": page_url,
			"format": fmt,
			"width": width,
			"height": height,
		}
		query_string = urlencode(params)
		return [f"{base}?{query_string}" for base in bases]

	async def _fetch_screenshot(self, shot_urls: list[str], fmt: str) -> tuple[bytes, str]:
		"""下载截图字节与 mime，支持多服务轮换重试。
		
		策略：每轮遍历所有配置的截图服务，失败后进入下一轮重试。
		例如：2个服务，2轮重试 = 服务1→服务2→服务1→服务2（最多4次尝试）
		"""
		mime = "image/webp" if fmt.lower() == "webp" else "image/png"
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		
		# 获取重试轮数配置（0=不重试只1轮，1-5=重试1-5轮）
		retry_rounds = int(self.config.get("screenshot_retry_rounds", 2))
		retry_rounds = max(0, min(5, retry_rounds))  # 限制在0-5之间
		total_rounds = retry_rounds + 1  # 首次尝试 + 重试轮数
		
		if not shot_urls:
			raise RuntimeError("没有配置截图服务")
		
		all_errors = []  # 记录所有失败
		total_attempts = 0
		
		# 多轮遍历所有服务
		for round_num in range(1, total_rounds + 1):
			if round_num > 1:
				logger.info(f"[webshot] ====== 开始第 {round_num}/{total_rounds} 轮重试 ======")
				await asyncio.sleep(1.5)  # 轮次间隔
			
			# 遍历所有截图服务
			for service_idx, shot_url in enumerate(shot_urls, 1):
				total_attempts += 1
				service_name = shot_url.split('//')[1].split('/')[0] if '//' in shot_url else shot_url[:30]
				
				if round_num == 1:
					logger.info(f"[webshot] 尝试服务 {service_idx}/{len(shot_urls)}: {service_name}")
				else:
					logger.info(f"[webshot] 第{round_num}轮 - 服务 {service_idx}/{len(shot_urls)}: {service_name}")
				
				try:
					async with httpx.AsyncClient(timeout=timeout) as client:
						resp = await client.get(shot_url)
						resp.raise_for_status()
						
						if total_attempts > 1:
							logger.info(f"[webshot] ✓ 截图成功！服务: {service_name}（第{round_num}轮第{service_idx}个服务）")
						return resp.content, mime
						
				except Exception as e:
					error_msg = f"第{round_num}轮-服务{service_idx}({service_name}): {str(e)[:80]}"
					all_errors.append(error_msg)
					logger.warning(f"[webshot] ✗ {error_msg}")
					await asyncio.sleep(0.5)  # 服务间短暂延迟
		
		# 所有轮次都失败了
		error_summary = "\n  ".join(all_errors[-6:])  # 显示最后6个错误
		raise RuntimeError(
			f"所有截图服务在 {total_rounds} 轮尝试中均失败（共 {total_attempts} 次尝试）。\n"
			f"最近的错误:\n  {error_summary}"
		)

	async def _fetch_page_text(self, url: str) -> Optional[str]:
		"""抓取网页并提取纯文本。"""
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		headers = {"User-Agent": self.config.get("fetch_user_agent", "Mozilla/5.0 AstrBot")}
		async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
			resp = await client.get(url)
			resp.raise_for_status()
			html = resp.text
			if BeautifulSoup is None:
				# 退化处理：简单去标签
				return html
			soup = BeautifulSoup(html, "html.parser")
			# 移除脚本与样式
			for tag in soup(["script", "style", "noscript"]):
				tag.decompose()
			text = soup.get_text("\n")
			# 简单压缩空行
			lines = [ln.strip() for ln in text.splitlines()]
			return "\n".join([ln for ln in lines if ln])

	def _make_image_part(self, image_bytes: bytes, mime: str):
		"""构造 Gemini SDK 所需的图片 Part。"""
		return types.Part.from_bytes(data=image_bytes, mime_type=mime)

	@staticmethod
	def _extract_text(resp) -> Optional[str]:
		"""兼容性提取：把 candidates/parts 文本拼起来。"""
		try:
			if not resp or not getattr(resp, "candidates", None):
				return None
			parts = []
			for c in resp.candidates:
				content = getattr(c, "content", None)
				if not content or not getattr(content, "parts", None):
					continue
				for p in content.parts:
					t = getattr(p, "text", None)
					if t:
						parts.append(t)
			return "\n".join(parts) if parts else None
		except Exception:
			return None

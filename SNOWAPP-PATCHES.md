# Snow App 适配补丁（非上游改动）

本仓库是 [xiphux/openai-api-bridge](https://github.com/xiphux/openai-api-bridge) 的 fork。
`snowapp` 分支承载**面向 Snow App 的本地适配** —— 这些改动**不打算提交上游**，
因为它们是客户端特有的适配，而非通用的桥增强。

`main` 分支保持与上游完全一致，未做任何改动。

---

## 补丁清单

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `src/openai_api_bridge/schemas/openai.py` | `response_format` 默认值 `url` → `b64_json` | 见下 |
| `src/openai_api_bridge/api/images.py` | 同上（`/v1/images/edits` 的 Form 默认值） | 见下 |

---

## 为什么需要这个改动

### 问题现象

Snow App 通过渠道调用桥生图时，图片**无法显示**，界面提示「远程图片加载失败」，
图片也不会写入 Snow App 的图像库。

### 根因链

1. 桥的鉴权**不在路由装饰器上**，而是 `auth.py` 里的 `ApiKeyMiddleware` ——
   一个最外层的 ASGI 中间件，**保护所有路由**（`auth.py` 开头原话：
   *"It is the bridge's only auth gate: every route requires the key"*）。

   注：v0.7.0 → v0.8.0 删除了 `/v1/files/{file_id}/content` 路由上的
   `dependencies=[Depends(require_api_key)]`，但那只是**清理冗余** ——
   鉴权依然由中间件承担。

2. 默认 `response_format=url` 时，桥返回的是
   `http://127.0.0.1:8398/v1/files/{id}/content` 这样的地址。

3. Snow App 拿到该 URL 后去下载，但**请求不带 `Authorization` 头**。

4. 中间件拦下 → `401 Unauthorized` → 图片永远拿不到。

### 解法

把默认 `response_format` 改成 `b64_json`：图片数据**随响应体内联返回**，
Snow App 无需发起第二次下载，问题自然消失。

代价是响应体变大（约 4/3 倍），但单张图 1~2 MB，远低于客户端上限。

---

## 另一个可选方案（未采用）

把 `/v1/files/*` 加入 `ApiKeyMiddleware` 的 `exempt_paths`，让该路由免鉴权。

未采用的原因：

- `exempt_paths` 是**精确匹配**（`scope["path"] in self.exempt_paths`），
  而文件路径是动态的 `/v1/files/{file_id}/content`，需要改成前缀匹配 ——
  对上游源码的侵入更大；
- 等于把图库资源对**本机任何进程**开放。

---

## 与上游同步

`main` 保持干净，同步上游只需：

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git checkout snowapp
git rebase main          # 或 git merge main
```

若上游未来自行支持了「免鉴权的资源 URL」或「客户端鉴权下载」，
本补丁即可退役。

---

## 部署说明

本分支配套的 Windows 原生运行配置不在本仓库内，见部署目录：

- `config.windows.toml` —— 原生运行的 provider 配置（`url` / `workflows_dir` 走 Windows 路径）
- `run-native.ps1` —— 启动脚本（从 `.env` 读 key，覆盖 `BRIDGE_CONFIG_PATH` / `FILES_DIR` / `SQLITE_PATH` / `PORT`）

启动后 `GET /v1/models` 会列出 `workflows/` 下每一对 `<name>.json` + `<name>.meta.json`，
以 `comfyui/<name>` 寻址。

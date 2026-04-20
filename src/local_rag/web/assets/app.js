const state = {
  uploading: false,
  querying: false,
  deletingChunk: false,
  switchingProvider: false,
  currentPage: 1,
  pageSize: 5,
  totalPages: 1,
};

const elements = {
  refreshButton: document.getElementById("refresh-status"),
  statusPill: document.getElementById("status-pill"),
  indexState: document.getElementById("index-state"),
  manifestSummary: document.getElementById("manifest-summary"),
  fileInput: document.getElementById("files-input"),
  fileList: document.getElementById("file-list"),
  sourceLabel: document.getElementById("source-label"),
  uploadForm: document.getElementById("upload-form"),
  uploadButton: document.getElementById("upload-button"),
  uploadResult: document.getElementById("upload-result"),
  queryForm: document.getElementById("query-form"),
  queryButton: document.getElementById("query-button"),
  questionInput: document.getElementById("question-input"),
  topKInput: document.getElementById("top-k-input"),
  answerOutput: document.getElementById("answer-output"),
  sourcesOutput: document.getElementById("sources-output"),
  generatorFlag: document.getElementById("generator-flag"),
  providerSelect: document.getElementById("provider-select"),
  applyProviderButton: document.getElementById("apply-provider-button"),
  refreshLlmStatusButton: document.getElementById("refresh-llm-status"),
  llmProviderValue: document.getElementById("llm-provider-value"),
  cudaStatusValue: document.getElementById("cuda-status-value"),
  localModelLoadedValue: document.getElementById("local-model-loaded-value"),
  localModelNameValue: document.getElementById("local-model-name-value"),
  llmStatusMessage: document.getElementById("llm-status-message"),
  pageSizeInput: document.getElementById("page-size-input"),
  prevPageButton: document.getElementById("prev-page-button"),
  nextPageButton: document.getElementById("next-page-button"),
  refreshChunksButton: document.getElementById("refresh-chunks"),
  chunksOutput: document.getElementById("chunks-output"),
  pageIndicator: document.getElementById("page-indicator"),
  chunksSummary: document.getElementById("chunks-summary"),
};

function setBusy(target, busy, idleText, busyText) {
  target.disabled = busy;
  target.textContent = busy ? busyText : idleText;
  if (busy) {
    target.classList.add("is-loading");
  } else {
    target.classList.remove("is-loading");
  }
}

function setStatus(indexReady) {
  elements.statusPill.textContent = indexReady ? "索引就绪" : "未建立索引";
  elements.indexState.textContent = indexReady ? "可检索" : "等待入库";
}

function formatManifest(manifest) {
  if (!manifest) {
    return "尚未建立知识库索引";
  }
  return `文档 ${manifest.document_count} 个 · 切片 ${manifest.chunk_count} 个 · 模型 ${manifest.embedding_model}`;
}

function renderLlmStatus(payload) {
  elements.providerSelect.value = payload.configured_provider;
  elements.llmProviderValue.textContent = payload.provider;
  elements.cudaStatusValue.textContent = payload.cuda_available
    ? "已检测到 CUDA"
    : "未检测到 CUDA";
  elements.localModelLoadedValue.textContent = payload.local_model_loaded
    ? "已加载"
    : "未加载";
  elements.localModelNameValue.textContent = payload.local_model_name || "-";
  elements.llmStatusMessage.textContent = payload.message || "状态正常";
}

function renderSources(matches) {
  if (!matches || matches.length === 0) {
    elements.sourcesOutput.textContent = "暂无结果";
    return;
  }

  elements.sourcesOutput.innerHTML = matches
    .map((match) => {
      const excerpt = (match.chunk.content || "").replace(/\s+/g, " ").trim();
      return `
        <article class="source-item">
          <div class="source-meta">
            <span>${escapeHtml(match.chunk.source_path)}</span>
            <span>score=${Number(match.score).toFixed(3)}</span>
          </div>
          <div class="source-text">${escapeHtml(excerpt)}</div>
        </article>
      `;
    })
    .join("");
}

function renderChunkPage(payload) {
  const items = payload?.items || [];
  if (items.length === 0) {
    elements.chunksOutput.textContent = "当前页没有数据。";
    return;
  }

  elements.chunksOutput.innerHTML = items
    .map(({ chunk }) => {
      const excerpt = (chunk.content || "").replace(/\s+/g, " ").trim();
      return `
        <article class="library-item">
          <div class="library-item-header">
            <div>
              <strong>${escapeHtml(chunk.source_path)}</strong>
              <span class="library-id">chunk_id=${escapeHtml(chunk.chunk_id)}</span>
            </div>
            <div class="library-actions">
              <span class="library-badge">#${Number(chunk.index) + 1} · ${Number(chunk.char_count)} chars</span>
              <button class="delete-button" type="button" data-chunk-id="${escapeHtml(chunk.chunk_id)}">删除</button>
            </div>
          </div>
          <div class="library-item-body">${escapeHtml(excerpt)}</div>
        </article>
      `;
    })
    .join("");

  bindDeleteButtons();
}

function updateChunkPagination(payload) {
  state.currentPage = payload.page;
  state.pageSize = payload.page_size;
  state.totalPages = payload.total_pages;
  elements.pageIndicator.textContent = `第 ${payload.page} / ${payload.total_pages} 页`;
  elements.chunksSummary.textContent = `共 ${payload.total} 条切片`;
  elements.prevPageButton.disabled = payload.page <= 1;
  elements.nextPageButton.disabled = payload.page >= payload.total_pages;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function refreshStatus() {
  try {
    const healthResponse = await fetch("/health");
    const healthPayload = await healthResponse.json();
    setStatus(Boolean(healthPayload.index_ready));

    if (!healthPayload.index_ready) {
      elements.manifestSummary.textContent = "尚未建立知识库索引";
      return;
    }

    const manifestResponse = await fetch("/manifest");
    if (!manifestResponse.ok) {
      throw new Error("无法读取 manifest");
    }
    const manifest = await manifestResponse.json();
    elements.manifestSummary.textContent = formatManifest(manifest);
    await refreshLlmStatus();
    await loadChunks(state.currentPage);
  } catch (error) {
    elements.statusPill.textContent = "状态异常";
    elements.indexState.textContent = "连接失败";
    elements.manifestSummary.textContent = error.message || "无法连接服务";
    elements.chunksOutput.textContent = error.message || "无法加载向量库内容";
    elements.llmStatusMessage.textContent =
      error.message || "无法读取 LLM 状态";
  }
}

async function refreshLlmStatus() {
  try {
    const response = await fetch("/llm/status");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "无法读取 LLM 状态");
    }
    renderLlmStatus(payload);
  } catch (error) {
    elements.llmStatusMessage.textContent =
      error.message || "无法读取 LLM 状态";
  }
}

async function handleProviderSwitch() {
  if (state.switchingProvider) {
    return;
  }

  state.switchingProvider = true;
  setBusy(elements.applyProviderButton, true, "切换 Provider", "切换中...");
  try {
    const response = await fetch("/llm/provider", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: elements.providerSelect.value }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "切换 provider 失败");
    }
    renderLlmStatus(payload);
  } catch (error) {
    elements.llmStatusMessage.textContent =
      error.message || "切换 provider 失败";
  } finally {
    state.switchingProvider = false;
    setBusy(elements.applyProviderButton, false, "切换 Provider", "切换中...");
  }
}

async function loadChunks(page = 1) {
  const pageSize = Number(elements.pageSizeInput.value || state.pageSize || 5);
  try {
    const response = await fetch(`/chunks?page=${page}&page_size=${pageSize}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "无法读取向量数据库内容");
    }

    renderChunkPage(payload);
    updateChunkPagination(payload);
  } catch (error) {
    elements.chunksOutput.textContent =
      error.message || "无法读取向量数据库内容";
    elements.chunksSummary.textContent = "暂无数据";
    elements.pageIndicator.textContent = "第 0 页";
  }
}

async function handleDeleteChunk(chunkId) {
  if (!chunkId || state.deletingChunk) {
    return;
  }

  const confirmed = window.confirm(`确认删除该切片？\n${chunkId}`);
  if (!confirmed) {
    return;
  }

  state.deletingChunk = true;
  elements.chunksSummary.textContent = "删除中...";

  try {
    const response = await fetch(`/chunks/${encodeURIComponent(chunkId)}`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "删除失败");
    }

    await refreshStatus();
    await loadChunks(state.currentPage);
  } catch (error) {
    elements.chunksSummary.textContent = error.message || "删除失败";
  } finally {
    state.deletingChunk = false;
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const files = Array.from(elements.fileInput.files || []);
  if (files.length === 0) {
    elements.uploadResult.textContent = "请先选择至少一个文件。";
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append(
    "source_label",
    elements.sourceLabel.value || "uploaded://web",
  );

  state.uploading = true;
  setBusy(elements.uploadButton, true, "上传并入库", "入库中...");
  elements.uploadResult.textContent = "正在上传并建立索引，请稍候...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "上传失败");
    }

    elements.uploadResult.textContent = JSON.stringify(payload, null, 2);
    await refreshStatus();
    await loadChunks(1);
  } catch (error) {
    elements.uploadResult.textContent = error.message || "上传失败";
  } finally {
    state.uploading = false;
    setBusy(elements.uploadButton, false, "上传并入库", "入库中...");
  }
}

async function handleQuery(event) {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  if (!question) {
    elements.answerOutput.textContent = "请输入问题。";
    return;
  }

  state.querying = true;
  setBusy(elements.queryButton, true, "开始检索问答", "检索中...");
  elements.answerOutput.textContent = "正在检索并生成回答...";
  elements.sourcesOutput.textContent = "正在读取命中片段...";

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(elements.topKInput.value || 4),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "查询失败");
    }

    elements.answerOutput.textContent = payload.answer || "未返回回答";
    elements.generatorFlag.textContent = payload.used_generator
      ? "LLM 已调用"
      : "摘要模式";
    renderSources(payload.matches);
  } catch (error) {
    elements.answerOutput.textContent = error.message || "查询失败";
    elements.sourcesOutput.textContent = "暂无结果";
    elements.generatorFlag.textContent = "未调用";
  } finally {
    state.querying = false;
    setBusy(elements.queryButton, false, "开始检索问答", "检索中...");
  }
}

function updateFileList() {
  const files = Array.from(elements.fileInput.files || []);
  if (files.length === 0) {
    elements.fileList.textContent = "尚未选择文件";
    return;
  }

  elements.fileList.textContent = files.map((file) => file.name).join(" / ");
}

function handlePrevPage() {
  if (state.currentPage > 1) {
    loadChunks(state.currentPage - 1);
  }
}

function handleNextPage() {
  if (state.currentPage < state.totalPages) {
    loadChunks(state.currentPage + 1);
  }
}

function handlePageSizeChange() {
  loadChunks(1);
}

function bindDeleteButtons() {
  document.querySelectorAll(".delete-button").forEach((button) => {
    button.addEventListener("click", () => {
      handleDeleteChunk(button.dataset.chunkId || "");
    });
  });
}

elements.refreshButton.addEventListener("click", refreshStatus);
elements.refreshLlmStatusButton.addEventListener("click", refreshLlmStatus);
elements.applyProviderButton.addEventListener("click", handleProviderSwitch);
elements.fileInput.addEventListener("change", updateFileList);
elements.uploadForm.addEventListener("submit", handleUpload);
elements.queryForm.addEventListener("submit", handleQuery);
elements.prevPageButton.addEventListener("click", handlePrevPage);
elements.nextPageButton.addEventListener("click", handleNextPage);
elements.refreshChunksButton.addEventListener("click", () =>
  loadChunks(state.currentPage),
);
elements.pageSizeInput.addEventListener("change", handlePageSizeChange);

refreshStatus();

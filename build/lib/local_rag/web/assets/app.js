const state = {
  uploading: false,
  querying: false,
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
  } catch (error) {
    elements.statusPill.textContent = "状态异常";
    elements.indexState.textContent = "连接失败";
    elements.manifestSummary.textContent = error.message || "无法连接服务";
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

elements.refreshButton.addEventListener("click", refreshStatus);
elements.fileInput.addEventListener("change", updateFileList);
elements.uploadForm.addEventListener("submit", handleUpload);
elements.queryForm.addEventListener("submit", handleQuery);

refreshStatus();

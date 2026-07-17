import fs from "node:fs/promises";
import path from "node:path";

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch (error) {
  throw new Error(
    "Playwright is required. Run `npm ci` and `npx playwright install chromium` first.",
    { cause: error },
  );
}

const [baseUrl, apiUrl, imagePath, outputRoot, audioPath, videoPath] = process.argv.slice(2);
if (!baseUrl || !apiUrl || !imagePath || !outputRoot) {
  throw new Error(
    "Usage: validate_multimodal_assistant.mjs BASE_URL API_URL IMAGE OUTPUT_ROOT [AUDIO] [VIDEO]",
  );
}
const expectedAssistantModel = process.env.VITALSSIGHT_EXPECTED_ASSISTANT_MODEL || "qwen3.6:35b";
const escapedAssistantModel = expectedAssistantModel.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const runRoot = path.resolve(outputRoot);
await fs.mkdir(runRoot, { recursive: true });

const checks = [];
const consoleErrors = [];
const pageErrors = [];
const responseErrors = [];

function check(name, condition, detail = "") {
  const passed = Boolean(condition);
  const item = { name, passed, detail: passed ? "" : detail };
  checks.push(item);
  if (!item.passed) throw new Error(`${name}: ${detail}`);
}

function attachDiagnostics(page) {
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("response", (response) => {
    if (response.status() >= 500) {
      responseErrors.push({ status: response.status(), url: response.url() });
    }
  });
}

async function bodyText(page) {
  return await page.locator("body").innerText();
}

async function waitForStreamlitIdle(page, timeout = 180000) {
  await page.waitForFunction(
    () => !Array.from(document.querySelectorAll("button")).some(
      (button) => button.offsetParent !== null && button.innerText.trim() === "Stop",
    ),
    undefined,
    { timeout },
  );
  await page.waitForTimeout(400);
}

async function waitForCountIncrease(locator, before, timeout = 360000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if ((await locator.count()) > before) return locator.nth(before);
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Expected locator count to exceed ${before} within ${timeout} ms.`);
}

async function openEnglish(page) {
  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.locator('[data-testid="stApp"]').waitFor({ state: "visible", timeout: 60000 });
  await waitForStreamlitIdle(page);
  if (await page.getByText("Evidence operations console", { exact: true }).first().isVisible().catch(() => false)) {
    return;
  }
  const expand = page.getByRole("button", { name: /keyboard_double_arrow_right/ });
  if (await expand.isVisible().catch(() => false)) await expand.click();
  const english = page.getByRole("radio", { name: "EN", exact: true });
  await english.waitFor({ state: "visible", timeout: 30000 });
  if (!(await english.isChecked())) await english.click({ force: true });
  await page.getByText("Evidence operations console", { exact: true }).first().waitFor({
    state: "visible",
    timeout: 30000,
  });
  await waitForStreamlitIdle(page);
}

async function gotoWorkspace(page, name) {
  const expand = page.getByRole("button", { name: /keyboard_double_arrow_right/ });
  if (await expand.isVisible().catch(() => false)) {
    await expand.click();
    await expand.waitFor({ state: "hidden", timeout: 10000 });
  }
  const radio = page.getByRole("radio", { name, exact: true });
  await radio.waitFor({ state: "attached", timeout: 15000 });
  await radio.locator("xpath=ancestor::label").click();
  await page.getByRole("heading", { name, exact: true }).first().waitFor({
    state: "visible",
    timeout: 30000,
  });
  await waitForStreamlitIdle(page);
}

async function findImageFileInput(page) {
  const inputs = page.locator('input[type="file"]');
  const count = await inputs.count();
  for (let index = 0; index < count; index += 1) {
    const input = inputs.nth(index);
    const accept = String((await input.getAttribute("accept")) || "").toLowerCase();
    if (accept.includes("image") || accept.includes(".png") || accept.includes(".jpg")) return input;
  }
  throw new Error(`No image file input found; observed ${count} file inputs.`);
}

async function findVideoFileInput(page) {
  const inputs = page.locator('input[type="file"]');
  const count = await inputs.count();
  for (let index = 0; index < count; index += 1) {
    const input = inputs.nth(index);
    const accept = String((await input.getAttribute("accept")) || "").toLowerCase();
    if (accept.includes("video") || accept.includes(".mp4") || accept.includes(".avi")) return input;
  }
  throw new Error(`No video file input found; observed ${count} file inputs.`);
}

async function ensureExpanderOpen(page) {
  const label = page.getByText("Unified AI workspace", { exact: true }).first();
  await label.waitFor({ state: "visible", timeout: 30000 });
  const details = label.locator("xpath=ancestor::details");
  if (await details.count()) {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      if (await details.evaluate((element) => element.open)) return;
      await label.click();
      await page.waitForTimeout(500);
    }
    throw new Error("Unified AI workspace expander did not remain open.");
  }
  await label.click();
  await page.waitForTimeout(500);
}

async function ensureVoicePanelOpen(page) {
  const voiceControl = page.getByText("Record a question or instruction", { exact: true }).first();
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await ensureExpanderOpen(page);
    const voiceTab = page.getByRole("tab", { name: "Voice to assistant", exact: true });
    if (await voiceTab.isVisible().catch(() => false)) await voiceTab.click();
    await page.waitForTimeout(500);
    if (await voiceControl.isVisible().catch(() => false)) return voiceControl;
  }
  throw new Error("Voice input panel did not remain visible after Streamlit redraws.");
}

const imageBuffer = await fs.readFile(path.resolve(imagePath));
const imageName = path.basename(imagePath);
const browser = await chromium.launch({
  headless: true,
  args: audioPath
    ? [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        `--use-file-for-fake-audio-capture=${path.resolve(audioPath)}`,
      ]
    : [],
});
const desktop = await browser.newContext({
  viewport: { width: 1440, height: 1050 },
  permissions: audioPath ? ["microphone"] : [],
});
const page = await desktop.newPage();
attachDiagnostics(page);
let mobile;

try {
  const apiHealthResponse = await desktop.request.get(`${apiUrl}/health`);
  check("API health endpoint responds", apiHealthResponse.ok(), apiHealthResponse.status());

  const assistantHealthResponse = await desktop.request.get(`${apiUrl}/api/v1/assistant/health`);
  check("assistant health endpoint responds", assistantHealthResponse.ok(), assistantHealthResponse.status());
  const assistantHealth = await assistantHealthResponse.json();
  check("text model is available", assistantHealth.model_available === true, JSON.stringify(assistantHealth));

  const multimodalHealthResponse = await desktop.request.get(`${apiUrl}/api/v1/assistant/multimodal/health`);
  check("multimodal health endpoint responds", multimodalHealthResponse.ok(), multimodalHealthResponse.status());
  const multimodalHealth = await multimodalHealthResponse.json();
  check("vision model is available", multimodalHealth.image?.available === true, JSON.stringify(multimodalHealth));
  check("speech model is available", multimodalHealth.speech?.available === true, JSON.stringify(multimodalHealth));

  const imageApiStarted = Date.now();
  const imageResponse = await desktop.request.post(`${apiUrl}/api/v1/assistant/analyze-image`, {
    multipart: {
      file: { name: imageName, mimeType: "image/png", buffer: imageBuffer },
      question: "Describe the visible authorized research image and the safest next workflow action without inferring identity or a vital sign.",
      language: "en",
    },
    timeout: 180000,
  });
  check("image analysis endpoint responds", imageResponse.ok(), `${imageResponse.status()} ${await imageResponse.text()}`);
  const imageResult = await imageResponse.json();
  check("image analysis is model-backed", imageResult.degraded === false, JSON.stringify(imageResult));
  check("raw image is not retained", imageResult.raw_image_retained === false, JSON.stringify(imageResult));
  check("image context is non-authoritative", imageResult.context?.authoritative === false, JSON.stringify(imageResult.context));
  checks.push({
    name: "image API latency",
    passed: true,
    detail: `${((Date.now() - imageApiStarted) / 1000).toFixed(2)} s`,
  });

  if (audioPath) {
    const audioBuffer = await fs.readFile(path.resolve(audioPath));
    const audioStarted = Date.now();
    const audioResponse = await desktop.request.post(`${apiUrl}/api/v1/assistant/transcribe`, {
      multipart: {
        file: { name: path.basename(audioPath), mimeType: "audio/wav", buffer: audioBuffer },
        language: "en",
      },
      timeout: 180000,
    });
    check("speech transcription endpoint responds", audioResponse.ok(), `${audioResponse.status()} ${await audioResponse.text()}`);
    const audioResult = await audioResponse.json();
    check("speech transcript is non-empty", Boolean(audioResult.transcript?.trim()), JSON.stringify(audioResult));
    check("raw audio is not retained", audioResult.raw_audio_retained === false, JSON.stringify(audioResult));
    checks.push({
      name: "speech API latency",
      passed: true,
      detail: `${((Date.now() - audioStarted) / 1000).toFixed(2)} s`,
    });
  }

  await openEnglish(page);
  const collapse = page.getByRole("button", { name: /keyboard_double_arrow_left/ });
  await collapse.waitFor({ state: "visible", timeout: 15000 });
  await collapse.click();
  const restore = page.getByRole("button", { name: /keyboard_double_arrow_right/ });
  await restore.waitFor({ state: "visible", timeout: 15000 });
  await restore.click();
  await page.getByText("Evidence operations console", { exact: true }).first().waitFor({ state: "visible", timeout: 15000 });
  check("desktop sidebar collapses and restores", true);

  await gotoWorkspace(page, "AI assistant");
  let text = await bodyText(page);
  check("assistant reports voice readiness", text.includes("Voice ready"), text.slice(0, 2400));
  check("assistant reports image readiness", text.includes("Image ready"), text.slice(0, 2400));
  check("assistant exposes controlled orchestration boundary", text.includes("AI orchestrates; evidence remains authoritative"), text.slice(0, 2400));

  const voiceControl = await ensureVoicePanelOpen(page);
  check("voice recorder control is present", true);
  if (audioPath) {
    const userMessages = page.getByLabel("Chat message from user");
    const assistantMessages = page.getByLabel("Chat message from assistant");
    const priorUserCount = await userMessages.count();
    const priorAssistantCount = await assistantMessages.count();
    const recordButton = page.getByRole("button", { name: "Record", exact: true });
    await recordButton.waitFor({ state: "visible", timeout: 15000 });
    await recordButton.click();
    const stopButton = page.getByRole("button", { name: "Stop recording", exact: true });
    await stopButton.waitFor({ state: "visible", timeout: 15000 });
    await page.waitForTimeout(6000);
    await stopButton.click();
    await waitForStreamlitIdle(page);
    await ensureVoicePanelOpen(page);
    const transcribeButton = page.getByRole("button", { name: /Transcribe and ask AI/ });
    await transcribeButton.waitFor({ state: "visible", timeout: 30000 });
    await transcribeButton.click();
    const transcript = page.getByRole("textbox", { name: "Review and edit transcript", exact: true });
    const voiceUserMessage = userMessages.nth(priorUserCount);
    await Promise.race([
      transcript.waitFor({ state: "visible", timeout: 180000 }),
      voiceUserMessage.waitFor({ state: "visible", timeout: 180000 }),
    ]);
    if (await transcript.isVisible().catch(() => false)) {
      check("uncertain browser transcript is editable", Boolean((await transcript.inputValue()).trim()));
      await page.getByRole("button", { name: /Confirm transcript and ask AI/ }).click();
    }
    await waitForCountIncrease(userMessages, priorUserCount, 180000);
    check("browser microphone recording reaches the assistant", Boolean((await voiceUserMessage.innerText()).trim()));
    await waitForCountIncrease(assistantMessages, priorAssistantCount, 360000);
    await ensureExpanderOpen(page);
  }
  await page.getByRole("tab", { name: "Image to assistant", exact: true }).click();
  const imageInput = await findImageFileInput(page);
  await imageInput.setInputFiles(path.resolve(imagePath));
  await waitForStreamlitIdle(page);
  await ensureExpanderOpen(page);
  await page.getByRole("tab", { name: "Image to assistant", exact: true }).click();
  const focusInput = page.getByRole("textbox", { name: "What should the assistant focus on?", exact: true });
  await focusInput.fill("Describe this authorized research image and what the user should do next without inferring identity or a vital sign.");
  const imageUserMessages = page.getByLabel("Chat message from user");
  const imageAssistantMessages = page.getByLabel("Chat message from assistant");
  const priorImageUserCount = await imageUserMessages.count();
  const priorImageAssistantCount = await imageAssistantMessages.count();
  const analyzeButton = page.getByRole("button", { name: /Analyze and ask AI/ });
  await analyzeButton.waitFor({ state: "visible", timeout: 30000 });
  await analyzeButton.click();
  const userMessage = await waitForCountIncrease(imageUserMessages, priorImageUserCount, 180000);
  await userMessage.getByText("Transient context: image context", { exact: true }).waitFor({ state: "visible", timeout: 30000 });
  const assistantMessage = await waitForCountIncrease(imageAssistantMessages, priorImageAssistantCount, 360000);
  await ensureExpanderOpen(page);
  await page.getByText("Visual summary", { exact: true }).waitFor({ state: "visible", timeout: 30000 });
  text = await bodyText(page);
  check("browser image analysis renders a summary", text.includes("Workflow relevance"), text.slice(-2800));
  check("browser explains raw-media retention", text.includes("Raw media is not retained"), text.slice(0, 2200));
  await page.screenshot({ path: path.join(runRoot, "multimodal_desktop_analysis.png"), fullPage: true });

  await assistantMessage
    .getByText(new RegExp(`Local model.*ollama.*${escapedAssistantModel}`, "i"))
    .waitFor({ state: "visible", timeout: 300000 });
  const answer = await assistantMessage.innerText();
  check("image context reaches the conversational assistant", /image|screen|workflow/i.test(answer), answer);
  check("assistant does not invent a vital-sign value", !/\b\d{2,3}(?:\.\d+)?\s*BPM\b/i.test(answer), answer);
  await page.screenshot({ path: path.join(runRoot, "multimodal_desktop_answer.png"), fullPage: true });

  if (videoPath) {
    await ensureExpanderOpen(page);
    await page.getByRole("tab", { name: "Video full workflow", exact: true }).click();
    const consent = page.getByRole("checkbox", {
      name: "I confirm this recording may be processed for the selected research purpose.",
      exact: true,
    });
    if (!(await consent.isChecked())) {
      await consent.check({ force: true });
      await waitForStreamlitIdle(page);
      await ensureExpanderOpen(page);
      await page.getByRole("tab", { name: "Video full workflow", exact: true }).click();
    }
    const videoInput = await findVideoFileInput(page);
    await videoInput.setInputFiles(path.resolve(videoPath));
    const videoInstruction = page.getByRole("textbox", { name: "What should the assistant return?", exact: true });
    await videoInstruction.fill("Explain the output state, evidence, next action and report boundary.");
    const videoAssistantMessages = page.getByLabel("Chat message from assistant");
    const priorVideoAssistantCount = await videoAssistantMessages.count();
    await page.getByRole("button", { name: /Run full workflow with AI/ }).click();
    await page.getByText("WORKFLOW COMPLETE", { exact: true }).waitFor({ state: "visible", timeout: 360000 });
    const videoAssistant = await waitForCountIncrease(videoAssistantMessages, priorVideoAssistantCount, 360000);
    const videoAnswer = await videoAssistant.innerText();
    text = await bodyText(page);
    check("video workflow returns an inline evidence report", text.includes("Inline evidence report"), text.slice(-5000));
    check("video workflow exposes report exports", /PDF[\s\S]*JSON[\s\S]*Markdown[\s\S]*CSV[\s\S]*FHIR/.test(text), text.slice(-5000));
    check("video workflow states raw video is not retained", text.includes("Raw video retained: no"), text.slice(-5000));
    if (/\bReview\b|\bRetake\b/.test(text)) {
      check("non-release video explanation withholds BPM", !/\b\d{2,3}(?:\.\d+)?\s*BPM\b/i.test(videoAnswer), videoAnswer);
    }
    await page.screenshot({ path: path.join(runRoot, "unified_video_workflow.png"), fullPage: true });
  }
  check("desktop viewport has no horizontal overflow", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2));

  mobile = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const mobilePage = await mobile.newPage();
  attachDiagnostics(mobilePage);
  await openEnglish(mobilePage);
  await gotoWorkspace(mobilePage, "AI assistant");
  await ensureVoicePanelOpen(mobilePage);
  check("mobile voice control remains visible", true);
  check("mobile viewport has no horizontal overflow", await mobilePage.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2));
  await mobilePage.screenshot({ path: path.join(runRoot, "multimodal_mobile.png"), fullPage: true });

  check("no browser console errors", consoleErrors.length === 0, JSON.stringify(consoleErrors));
  check("no browser page errors", pageErrors.length === 0, JSON.stringify(pageErrors));
  check("no HTTP 5xx responses", responseErrors.length === 0, JSON.stringify(responseErrors));

  const report = {
    schema_version: "vitalssight.multimodal-browser-validation.v1",
    passed: checks.every((item) => item.passed),
    base_url: baseUrl,
    api_url: apiUrl,
    image_fixture: imageName,
    audio_fixture: audioPath ? path.basename(audioPath) : null,
    video_fixture: videoPath ? path.basename(videoPath) : null,
    multimodal_health: multimodalHealth,
    checks,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    response_errors: responseErrors,
    claim_boundary: "Finite local product conformance; not clinical validation, diagnostic evidence, or production security certification.",
  };
  await fs.writeFile(path.join(runRoot, "multimodal_validation.json"), JSON.stringify(report, null, 2), "utf8");
  await fs.writeFile(
    path.join(runRoot, "multimodal_validation.md"),
    [
      "# VitalsSight multimodal assistant validation",
      "",
      `- Result: ${report.passed ? "PASS" : "FAIL"}`,
      `- Checks: ${checks.filter((item) => item.passed).length}/${checks.length}`,
      `- Vision model: ${multimodalHealth.image?.model || "unknown"}`,
      `- Speech model: ${multimodalHealth.speech?.model || "unknown"}`,
      "",
      ...checks.map((item) => `- [${item.passed ? "x" : " "}] ${item.name}${item.detail ? `: ${item.detail}` : ""}`),
      "",
      `Boundary: ${report.claim_boundary}`,
      "",
    ].join("\n"),
    "utf8",
  );
  console.log(JSON.stringify({ passed: true, checks: checks.length, output: runRoot }, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(runRoot, "multimodal_failure.png"), fullPage: true }).catch(() => {});
  await fs.writeFile(path.join(runRoot, "multimodal_failure.txt"), await bodyText(page), "utf8").catch(() => {});
  const report = {
    schema_version: "vitalssight.multimodal-browser-validation.v1",
    passed: false,
    error: String(error?.stack || error),
    checks,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    response_errors: responseErrors,
  };
  await fs.writeFile(path.join(runRoot, "multimodal_validation.json"), JSON.stringify(report, null, 2), "utf8");
  throw error;
} finally {
  await mobile?.close();
  await desktop.close();
  await browser.close();
}

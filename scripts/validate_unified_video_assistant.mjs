import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const [baseUrl, videoPath, outputRoot] = process.argv.slice(2);
if (!baseUrl || !videoPath || !outputRoot) {
  throw new Error("Usage: validate_unified_video_assistant.mjs BASE_URL VIDEO OUTPUT_ROOT");
}

const runRoot = path.resolve(outputRoot);
await fs.mkdir(runRoot, { recursive: true });
const checks = [];
const consoleErrors = [];
const pageErrors = [];
const responseErrors = [];

function check(name, condition, detail = "") {
  const passed = Boolean(condition);
  checks.push({ name, passed, detail: passed ? "" : detail });
  if (!passed) throw new Error(`${name}: ${detail}`);
}

async function waitForIdle(page, timeout = 360000) {
  await page.waitForFunction(
    () => !Array.from(document.querySelectorAll("button")).some(
      (button) => button.offsetParent !== null && button.innerText.trim() === "Stop",
    ),
    undefined,
    { timeout },
  );
  await page.waitForTimeout(500);
}

async function openEnglish(page) {
  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.locator('[data-testid="stApp"]').waitFor({ state: "visible", timeout: 60000 });
  await waitForIdle(page);
  const expand = page.getByRole("button", { name: /keyboard_double_arrow_right/ });
  if (await expand.isVisible().catch(() => false)) await expand.click();
  const english = page.getByRole("radio", { name: "EN", exact: true });
  await english.waitFor({ state: "visible", timeout: 30000 });
  if (!(await english.isChecked())) await english.click({ force: true });
  await waitForIdle(page);
}

async function gotoAssistant(page) {
  const radio = page.getByRole("radio", { name: "AI assistant", exact: true });
  await radio.waitFor({ state: "visible", timeout: 30000 });
  await radio.click({ force: true });
  await page.getByRole("heading", { name: "AI assistant", exact: true }).first().waitFor({
    state: "visible",
    timeout: 30000,
  });
  await waitForIdle(page);
}

async function findVideoInput(page) {
  const inputs = page.locator('input[type="file"]');
  for (let index = 0; index < await inputs.count(); index += 1) {
    const input = inputs.nth(index);
    const accept = String((await input.getAttribute("accept")) || "").toLowerCase();
    if (accept.includes("video") || accept.includes(".avi") || accept.includes(".mp4")) return input;
  }
  throw new Error("Video upload input was not found.");
}

async function waitForNewMessage(locator, previousCount, timeout = 360000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await locator.count() > previousCount) return locator.nth(previousCount);
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`No new chat message appeared within ${timeout} ms.`);
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
const page = await context.newPage();
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => pageErrors.push(String(error)));
page.on("response", (response) => {
  if (response.status() >= 500) responseErrors.push({ status: response.status(), url: response.url() });
});

try {
  await openEnglish(page);
  await gotoAssistant(page);
  await page.getByText("Unified AI workspace", { exact: true }).first().waitFor({ state: "visible" });
  await page.getByRole("tab", { name: "Video full workflow", exact: true }).click();

  let consent = page.getByRole("checkbox", {
    name: "I confirm this recording may be processed for the selected research purpose.",
    exact: true,
  });
  if (!(await consent.isChecked())) {
    await consent.check({ force: true });
    await waitForIdle(page);
    await page.getByRole("tab", { name: "Video full workflow", exact: true }).click();
    consent = page.getByRole("checkbox", {
      name: "I confirm this recording may be processed for the selected research purpose.",
      exact: true,
    });
  }
  check("video consent is explicitly recorded", await consent.isChecked());
  await (await findVideoInput(page)).setInputFiles(path.resolve(videoPath));
  await page.getByRole("textbox", { name: "What should the assistant return?", exact: true }).fill(
    "Explain the output state, evidence, next action and report boundary.",
  );
  const assistantMessages = page.getByLabel("Chat message from assistant");
  const previousAssistantCount = await assistantMessages.count();
  await page.getByRole("button", { name: /Run full workflow with AI/ }).click();

  await page.getByText("WORKFLOW COMPLETE", { exact: true }).waitFor({ state: "visible", timeout: 360000 });
  const assistantMessage = await waitForNewMessage(assistantMessages, previousAssistantCount);
  const assistantText = await assistantMessage.innerText();
  const body = await page.locator("body").innerText();
  check("deterministic video workflow returns retake", /\bRetake\b/.test(body), body.slice(-6000));
  check("inline evidence report is rendered", body.includes("Inline evidence report"), body.slice(-6000));
  check("all inline exports are rendered", /PDF[\s\S]*JSON[\s\S]*Markdown[\s\S]*CSV[\s\S]*FHIR/.test(body), body.slice(-6000));
  check("raw video is not retained", body.includes("Raw video retained: no"), body.slice(-6000));
  check("retake assistant response contains no BPM", !/\b\d{2,3}(?:\.\d+)?\s*BPM\b/i.test(assistantText), assistantText);
  check("desktop has no horizontal overflow", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2));
  check("no browser console errors", consoleErrors.length === 0, JSON.stringify(consoleErrors));
  check("no browser page errors", pageErrors.length === 0, JSON.stringify(pageErrors));
  check("no HTTP 5xx responses", responseErrors.length === 0, JSON.stringify(responseErrors));
  await page.screenshot({ path: path.join(runRoot, "unified_video_workflow.png"), fullPage: true });

  const report = {
    schema_version: "vitalssight.unified-video-assistant-validation.v1",
    passed: checks.every((item) => item.passed),
    video_fixture: path.basename(videoPath),
    checks,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    response_errors: responseErrors,
    claim_boundary: "Finite research-product conformance; not clinical validation or a medical-device claim.",
  };
  await fs.writeFile(path.join(runRoot, "validation.json"), JSON.stringify(report, null, 2), "utf8");
  console.log(JSON.stringify({ passed: report.passed, checks: checks.length, output: runRoot }, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(runRoot, "failure.png"), fullPage: true }).catch(() => {});
  await fs.writeFile(path.join(runRoot, "failure.txt"), await page.locator("body").innerText(), "utf8").catch(() => {});
  await fs.writeFile(
    path.join(runRoot, "validation.json"),
    JSON.stringify({ passed: false, error: String(error?.stack || error), checks, consoleErrors, pageErrors, responseErrors }, null, 2),
    "utf8",
  );
  throw error;
} finally {
  await context.close();
  await browser.close();
}

import fs from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch (error) {
  throw new Error(
    "Playwright is required for browser validation. Run `npm ci` and " +
      "`npx playwright install chromium` before invoking this script.",
    { cause: error },
  );
}

const [baseUrl, apiUrl, fixtureRoot, outputRoot, commit] = process.argv.slice(2);
if (!baseUrl || !apiUrl || !fixtureRoot || !outputRoot || !commit) {
  throw new Error("Usage: validate_browser_product.mjs BASE_URL API_URL FIXTURE_ROOT OUTPUT_ROOT COMMIT");
}

const execFileAsync = promisify(execFile);
const actualCommit = (await execFileAsync("git", ["rev-parse", "HEAD"])).stdout.trim();
const gitTree = (await execFileAsync("git", ["rev-parse", "HEAD^{tree}"])).stdout.trim();
const workingTreeStatus = (await execFileAsync("git", ["status", "--porcelain"])).stdout.trim();
if (commit !== actualCommit) {
  throw new Error(`Expected commit ${commit}, but the checked-out commit is ${actualCommit}.`);
}
if (workingTreeStatus) {
  throw new Error(`Browser validation requires a clean working tree:\n${workingTreeStatus}`);
}

const finalRun = path.join(outputRoot, "playwright", "final_run");
const downloadsDir = path.join(outputRoot, "playwright", "downloads");
await fs.mkdir(finalRun, { recursive: true });
await fs.mkdir(downloadsDir, { recursive: true });

const consoleErrors = [];
const pageErrors = [];
const responseErrors = [];
const checks = [];

function attachDiagnostics(page) {
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("response", (response) => {
    if (response.status() >= 400) responseErrors.push({ status: response.status(), url: response.url() });
  });
}

function check(name, condition, detail = "") {
  checks.push({ name, passed: Boolean(condition), detail });
  if (!condition) throw new Error(`${name}: ${detail}`);
}

async function bodyText(page) {
  return await page.locator("body").innerText();
}

async function saveState(page, stem) {
  await page.screenshot({ path: path.join(finalRun, `${stem}.png`), fullPage: true });
  await fs.writeFile(path.join(finalRun, `${stem}.txt`), await bodyText(page), "utf8");
}

async function regularFiles(root) {
  const files = [];
  async function visit(current) {
    let entries;
    try {
      entries = await fs.readdir(current, { withFileTypes: true });
    } catch (error) {
      if (error?.code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const target = path.join(current, entry.name);
      if (entry.isDirectory()) await visit(target);
      else if (entry.isFile()) files.push(target);
    }
  }
  await visit(root);
  return files;
}

async function waitForHeading(page, name, timeout = 30000) {
  await page.getByRole("heading", { name, exact: true }).first().waitFor({ state: "visible", timeout });
  await page.getByRole("button", { name: "Stop", exact: true }).waitFor({ state: "hidden", timeout: 120000 });
  await page.waitForTimeout(300);
}

async function gotoWorkspace(page, name) {
  const radio = page.getByRole("radio", { name, exact: true });
  const expand = page.getByRole("button", { name: /keyboard_double_arrow_right/ });
  if (await expand.isVisible().catch(() => false)) {
    // Let the one-shot mobile navigation close finish before testing reopen.
    await page.waitForTimeout(300);
    await expand.click();
    await expand.waitFor({ state: "hidden", timeout: 10000 });
    await page.waitForTimeout(300);
  }
  const option = radio.locator("xpath=ancestor::label");
  await option.scrollIntoViewIfNeeded();
  await option.click();
  await waitForHeading(page, name);
}

async function openEnglish(page) {
  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.getByRole("heading", { name: /总览|Overview/ }).first().waitFor({ state: "visible", timeout: 60000 });
  await page.getByRole("button", { name: "Stop", exact: true }).waitFor({ state: "hidden", timeout: 120000 });
  const englishShell = page.getByText("Evidence operations console", { exact: true }).first();
  if (await englishShell.isVisible().catch(() => false)) {
    check("English language control is active", true);
    await waitForHeading(page, "Overview");
    return;
  }
  const expand = page.getByRole("button", { name: /keyboard_double_arrow_right/ });
  if (await expand.isVisible().catch(() => false)) {
    await page.waitForTimeout(300);
    await expand.click();
    await expand.waitFor({ state: "hidden", timeout: 10000 });
  }
  const en = page.getByRole("radio", { name: "EN", exact: true });
  await en.waitFor({ state: "visible", timeout: 30000 });
  if (!(await en.isChecked())) {
    await en.click({ force: true });
    await page.getByText("Evidence operations console", { exact: true }).first().waitFor({ state: "visible", timeout: 30000 });
    await en.waitFor({ state: "visible", timeout: 30000 });
  }
  check("English language control is active", await en.isChecked());
  await waitForHeading(page, "Overview");
}

async function prepareUpload(page, fixture) {
  await gotoWorkspace(page, "New assessment");
  const consent = page.getByRole("checkbox", {
    name: "I confirm the recording may be processed for the selected research purpose.",
  });
  if (!(await consent.isChecked())) {
    await consent.click({ force: true });
    await page.waitForTimeout(800);
  }
  check("consent control accepts confirmation", await page.getByRole("checkbox", {
    name: "I confirm the recording may be processed for the selected research purpose.",
  }).isChecked());
  await page.getByRole("radio", { name: "Upload video", exact: true }).click({ force: true });
  await page.locator('input[type="file"]').first().waitFor({ state: "attached", timeout: 15000 });
  const input = page.locator('input[type="file"]').first();
  await input.setInputFiles(path.join(fixtureRoot, fixture));
  await page.getByRole("button", { name: /Remove / }).waitFor({ state: "visible", timeout: 30000 });
}

async function runAssessment(page, fixture, expected) {
  await prepareUpload(page, fixture);
  await page.getByRole("button", { name: /Run assessment/ }).click();
  await page.getByText(new RegExp(`Assessment completed: ${expected}`, "i")).first().waitFor({
    state: "visible",
    timeout: 240000,
  });
  await page.waitForFunction(
    (state) => {
      const text = document.body.innerText;
      const outputReady = !text.includes("No result has been generated in this session.");
      const expectedOutput = state === "Release"
        ? text.includes("Published HR") && text.includes("75.1 BPM")
        : text.includes("Published HR") && text.includes("Withheld");
      return outputReady && expectedOutput;
    },
    expected,
    { timeout: 240000 },
  );
  await page.waitForTimeout(500);
  const text = await bodyText(page);
  check(`${expected} state visible`, text.includes(expected), text.slice(-1200));
  if (expected === "Release") {
    check("release publishes finite HR", /Published HR\s+75\.1 BPM/.test(text), text.slice(-1400));
  } else {
    check(`${expected} withholds HR`, /Published HR\s+Withheld/.test(text), text.slice(-1400));
  }
  return text;
}

async function downloadByButton(page, buttonName, expectedExtension) {
  const downloadPromise = page.waitForEvent("download", { timeout: 30000 });
  await page.getByRole("button", { name: new RegExp(buttonName) }).first().click();
  const download = await downloadPromise;
  const suggested = download.suggestedFilename();
  check(`${buttonName} extension`, suggested.toLowerCase().endsWith(expectedExtension), suggested);
  const target = path.join(downloadsDir, suggested);
  await download.saveAs(target);
  const stat = await fs.stat(target);
  check(`${buttonName} non-empty`, stat.size > 20, `${stat.size} bytes`);
  return target;
}

async function chooseComboboxOption(page, combobox, optionName) {
  await combobox.waitFor({ state: "visible", timeout: 10000 });
  await combobox.focus();
  await combobox.press("ArrowDown");
  const option = page.getByRole("option", { name: optionName, exact: true });
  await option.waitFor({ state: "visible", timeout: 10000 });
  await option.click();
}

const browser = await chromium.launch({
  headless: true,
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  acceptDownloads: true,
});
const page = await context.newPage();
attachDiagnostics(page);
let mobileContext;

try {
  await openEnglish(page);
  const healthResponse = await context.request.get(`${apiUrl}/health`);
  check("API health endpoint responds", healthResponse.ok(), healthResponse.status());
  const health = await healthResponse.json();
  check("API service commit matches validation commit", health.build?.commit === actualCommit, health.build?.commit);
  check("API service tree matches validation tree", health.build?.tree === gitTree, health.build?.tree);
  check("API service reports a clean source tree", health.build?.dirty === false, health.build?.dirty);
  const buildLabel = `Build ${actualCommit.slice(0, 12)} · Tree ${gitTree.slice(0, 12)} · clean`;
  check("Streamlit service build identity matches validation source", (await bodyText(page)).includes(buildLabel), buildLabel);
  check("desktop viewport has no horizontal overflow", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2));
  await saveState(page, "01_overview_desktop");

  const collapse = page.getByRole("button", { name: /keyboard_double_arrow_left/ });
  await collapse.click();
  await page.getByRole("button", { name: /keyboard_double_arrow_right/ }).waitFor({ state: "visible", timeout: 10000 });
  check("collapsed sidebar can be restored", true);
  await page.getByRole("button", { name: /keyboard_double_arrow_right/ }).click();
  await page.getByText("Evidence operations console", { exact: true }).first().waitFor({ state: "visible", timeout: 10000 });
  await saveState(page, "02_sidebar_restored");

  await gotoWorkspace(page, "New assessment");
  await page.getByRole("button", { name: /Run assessment/ }).click();
  await page.getByTestId("stAlertContentWarning").getByText(
    "Confirm processing consent before running the assessment.",
    { exact: true },
  ).waitFor({ state: "visible", timeout: 15000 });
  check("unconsented assessment returns an explicit warning", true);
  check("unconsented assessment produces no output", (await bodyText(page)).includes("No result has been generated in this session."));

  let text = await runAssessment(page, "8555_IriunWebcam_before.avi", "Release");
  check("release evidence action present", text.includes("Retain the evidence packet"));
  await saveState(page, "03_real_video_release");
  await page.getByRole("button", { name: /Clear/ }).click();
  await page.getByText("Assessment input and session result were cleared.").first().waitFor({ state: "visible", timeout: 15000 });
  const clearedUploader = page.locator('input[type="file"]').first();
  await clearedUploader.waitFor({ state: "attached", timeout: 15000 });
  check(
    "clear rebuilds an empty file uploader",
    await clearedUploader.evaluate((element) => element.files?.length === 0),
  );
  check("clear removes the prior filename", !(await bodyText(page)).includes("8555_Ir...before.avi"));
  await saveState(page, "04_clear_after_release");

  text = await runAssessment(page, "1285_USBVideo_before.avi", "Review");
  check(
    "review reason is translated into an actionable user explanation",
    text.includes("Keep the competing tracks linked to the case and route them to review."),
  );
  await saveState(page, "05_real_video_review");
  await gotoWorkspace(page, "Review queue");
  check("review queue exposes withheld output", (await bodyText(page)).includes("Withheld"));
  const reviewForm = page.getByTestId("stForm");
  const assignee = reviewForm.getByRole("textbox", { name: "Assignee" });
  await assignee.fill("Browser validation reviewer");
  const note = reviewForm.getByRole("textbox", { name: "Reviewer note" });
  await note.fill("Real-video browser replay: competing tracks inspected; output remains withheld for research review.");
  await reviewForm.getByRole("button", { name: /Save review/ }).click();
  await page.getByText("Review record saved with an audit event.").first().waitFor({ state: "visible", timeout: 20000 });
  await chooseComboboxOption(
    page,
    reviewForm.getByRole("combobox", { name: "Status", exact: true }),
    "in_review",
  );
  await chooseComboboxOption(
    page,
    reviewForm.getByRole("combobox", { name: "Resolution", exact: true }),
    "retain_for_research_review",
  );
  await reviewForm.getByRole("button", { name: /Save review/ }).click();
  await page.getByText("Review record saved with an audit event.").first().waitFor({ state: "visible", timeout: 20000 });
  const renderedResolutionValues = await page
    .getByRole("combobox", { name: "Resolution", exact: true })
    .evaluateAll((elements) => elements.map((element) => element.value));
  check(
    "review resolution persists after rerender",
    renderedResolutionValues.length > 0
      && renderedResolutionValues.every((value) => value === "retain_for_research_review"),
    JSON.stringify(renderedResolutionValues),
  );
  await page.getByText("Audit trail", { exact: true }).first().click();
  const reviewAuditCell = page.getByText("review.updated", { exact: true });
  await reviewAuditCell.first().waitFor({ state: "attached", timeout: 15000 });
  text = await bodyText(page);
  check("review audit event is rendered in the audit grid", await reviewAuditCell.count() > 0);
  await saveState(page, "06_review_saved_with_audit");

  text = await runAssessment(page, "8555_retake_first5s.avi", "Retake");
  check("retake recommends duration correction", text.includes("Record at least 8 seconds; 20-30 seconds is preferred."));
  check("retake does not invent illumination correction", !text.includes("Use one even, front-facing light source and avoid backlight"), text.slice(-1800));
  check("retake does not invent candidate-count correction", !text.includes("Confirm at least three candidates are retained"), text.slice(-1800));
  await saveState(page, "07_real_video_retake_corrected_guidance");

  await page.getByRole("button", { name: /Build report/ }).click();
  await waitForHeading(page, "Reports");
  await page.getByRole("button", { name: /Report PDF/ }).waitFor({ state: "visible", timeout: 60000 });
  await page.getByText("Implementation provenance", { exact: true }).first().waitFor({ state: "visible", timeout: 30000 });
  text = await bodyText(page);
  check("retake report page exposes implementation provenance", text.includes("Implementation provenance"));
  for (const [tabName, expectedText] of [
    ["Evidence to action", "WHY THIS ACTION"],
    ["Attribution", "Source field"],
    ["Review & audit", "Reviewer note"],
    ["Structured data", "report_version"],
  ]) {
    await page.getByRole("tab", { name: tabName, exact: true }).click();
    await page.waitForTimeout(150);
    check(`report tab renders: ${tabName}`, (await bodyText(page)).includes(expectedText), expectedText);
  }
  await page.getByRole("tab", { name: "Report detail", exact: true }).click();
  await saveState(page, "08_retake_report_detail");

  const pdfPath = await downloadByButton(page, "Report PDF", ".pdf");
  const jsonPath = await downloadByButton(page, "Evidence JSON", ".json");
  const markdownPath = await downloadByButton(page, "Review Markdown", ".md");
  const csvPath = await downloadByButton(page, "Case CSV", ".csv");
  const pdfHeader = (await fs.readFile(pdfPath)).subarray(0, 4).toString("ascii");
  check("downloaded report is PDF", pdfHeader === "%PDF", pdfHeader);
  const reportJson = JSON.parse(await fs.readFile(jsonPath, "utf8"));
  check("downloaded JSON retains retake", reportJson.case.decision === "retake", reportJson.case.decision);
  check(
    "downloaded JSON contains no absolute local filesystem path",
    !/[A-Za-z]:[\\/]/.test(JSON.stringify(reportJson)) && !/\/(?:home|Users)\//.test(JSON.stringify(reportJson)),
  );
  check(
    "downloaded JSON retains passing illumination evidence",
    reportJson.case.preflight.checks.some((item) => item.check === "illumination" && item.status === "pass"),
  );
  const reportMarkdown = await fs.readFile(markdownPath, "utf8");
  check("downloaded Markdown has corrected preflight chain", reportMarkdown.includes("Candidate construction | not entered | not evaluated"));
  check("downloaded Markdown excludes false illumination failure", !reportMarkdown.includes("Illumination score: 41%"));
  check("downloaded CSV has case row", (await fs.readFile(csvPath, "utf8")).split(/\r?\n/).length >= 2);

  const workspaces = ["Overview", "New assessment", "Cases", "Review queue", "Reports", "Evidence", "Integrations", "Help & settings"];
  for (const workspace of workspaces) {
    await gotoWorkspace(page, workspace);
    check(`workspace opens: ${workspace}`, true);
  }
  await saveState(page, "09_help_and_settings");
  text = await bodyText(page);
  check("role-based guide names required input action output and next destination", text.includes("Each step states the required input"));
  check("capture guide exposes a complete start action", await page.getByRole("button", { name: "Start this workflow", exact: true }).isVisible());
  await page.getByRole("button", { name: "Start this workflow", exact: true }).click();
  await waitForHeading(page, "New assessment");
  await page.getByText("Guided assessment opened at purpose and consent.", { exact: true }).first().waitFor({ state: "visible", timeout: 15000 });
  check("guided workflow start action navigates to assessment input", true);
  await gotoWorkspace(page, "Help & settings");
  const operator = page.getByRole("textbox", { name: "Operator name" });
  await operator.fill("Browser QA operator");
  await page.getByRole("button", { name: /Save operator/ }).click();
  await page.getByText("Operator saved for future audit events.").first().waitFor({ state: "visible", timeout: 15000 });
  check("operator setting saved", true);

  await gotoWorkspace(page, "Integrations");
  await page.getByRole("button", { name: /Write integration audit event/ }).click();
  await page.getByText("Audit event recorded for this payload.").first().waitFor({ state: "visible", timeout: 15000 });
  check("integration audit action gives visible feedback", true);
  const openapiPath = await downloadByButton(page, "OpenAPI schema", ".json");
  const openapi = JSON.parse(await fs.readFile(openapiPath, "utf8"));
  check("OpenAPI contains video assessment endpoint", Boolean(openapi.paths?.["/api/v1/assessments/video"]));
  await saveState(page, "10_integrations_audit");
  const integratedReportResponse = await context.request.get(
    `${apiUrl}/api/v1/cases/${reportJson.case.case_id}/report?format=json`,
  );
  check("integration audit report endpoint responds", integratedReportResponse.ok(), integratedReportResponse.status());
  const integratedReport = await integratedReportResponse.json();
  check(
    "integration audit event persists in the shared case store",
    integratedReport.audit_events.some((event) => event.event_type === "integration.payload_validated"),
  );
  await gotoWorkspace(page, "Help & settings");
  await page.getByRole("button", { name: "Restore built-in demo cases", exact: true }).click();
  await page.getByText("Built-in cases restored without deleting user cases.", { exact: true }).first().waitFor({ state: "visible", timeout: 15000 });
  const restoredCasesResponse = await context.request.get(`${apiUrl}/api/v1/cases`);
  check("restored case registry endpoint responds", restoredCasesResponse.ok(), restoredCasesResponse.status());
  const restoredCases = await restoredCasesResponse.json();
  check(
    "demo restoration preserves the real uploaded case",
    restoredCases.items.some((item) => item.source_name === "8555_retake_first5s.avi"),
  );

  const docs = await context.newPage();
  await docs.goto(`${apiUrl}/docs`, { waitUntil: "domcontentloaded", timeout: 30000 });
  await docs.getByText("VitalsSight Evidence API", { exact: false }).first().waitFor({ state: "visible", timeout: 30000 });
  await docs.screenshot({ path: path.join(finalRun, "11_api_docs.png"), fullPage: true });
  await docs.close();
  check("interactive API documentation loads", true);

  mobileContext = await browser.newContext({
    viewport: { width: 390, height: 844 },
    acceptDownloads: true,
  });
  const mobilePage = await mobileContext.newPage();
  attachDiagnostics(mobilePage);
  await openEnglish(mobilePage);
  check("fresh mobile session renders the sidebar", await mobilePage.getByText("Evidence operations console", { exact: true }).first().isVisible());
  const initialCollapse = mobilePage.getByRole("button", { name: /keyboard_double_arrow_left/ });
  await initialCollapse.click();
  const initialExpand = mobilePage.getByRole("button", { name: /keyboard_double_arrow_right/ });
  await initialExpand.waitFor({ state: "visible", timeout: 10000 });
  await initialExpand.click();
  await mobilePage.getByText("Evidence operations console", { exact: true }).first().waitFor({ state: "visible", timeout: 10000 });
  check("fresh mobile sidebar can collapse and reopen", true);
  await gotoWorkspace(mobilePage, "Integrations");
  const mobileExpand = mobilePage.getByRole("button", { name: /keyboard_double_arrow_right/ });
  check("mobile integration navigation auto-closes sidebar", await mobileExpand.isVisible());
  await gotoWorkspace(mobilePage, "New assessment");
  check("mobile navigation auto-closes sidebar", await mobilePage.getByRole("button", { name: /keyboard_double_arrow_right/ }).isVisible());
  check("mobile viewport has no horizontal overflow", await mobilePage.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2));
  await saveState(mobilePage, "12_mobile_new_assessment");

  check(
    "tested delete-after-analysis mode leaves no raw upload file",
    (await regularFiles(path.join(outputRoot, "uploads"))).length === 0,
  );

  check("no browser console errors", consoleErrors.length === 0, JSON.stringify(consoleErrors));
  check("no page errors", pageErrors.length === 0, JSON.stringify(pageErrors));
  check("no unexpected HTTP response errors", responseErrors.length === 0, JSON.stringify(responseErrors));

  const manifest = {
    validation_version: "vitalssight.browser-product-validation.v3",
    passed: checks.every((item) => item.passed),
    git_commit: actualCommit,
    git_tree: gitTree,
    working_tree_clean: true,
    service_build: health.build,
    base_url: baseUrl,
    api_url: apiUrl,
    viewports: ["1440x1000", "390x844"],
    real_video_cases: {
      release: "8555_IriunWebcam_before.avi",
      review: "1285_USBVideo_before.avi",
      retake: "8555_retake_first5s.avi",
    },
    checks,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    response_errors: responseErrors,
    claim_boundary: "Finite browser and API conformance on curated fixtures; not clinical validation, usability evidence, security certification, or production readiness.",
  };
  await fs.writeFile(path.join(outputRoot, "browser_validation_manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  await fs.writeFile(
    path.join(outputRoot, "BROWSER_VALIDATION_SUMMARY.md"),
    [
      "# VitalsSight browser validation",
      "",
      `Overall result: ${manifest.passed ? "PASS" : "FAIL"}`,
      "",
      `- Git commit: \`${actualCommit}\``,
      `- Git tree: \`${gitTree}\``,
      "- Working tree: clean",
      `- Checks: ${checks.filter((item) => item.passed).length}/${checks.length}`,
      "- Real-video states: release, review, retake",
      "- Reports: PDF, JSON, Markdown, CSV",
      "- Workspaces: 8/8",
      "- Viewports: 1440x1000 and 390x844",
      `- Console errors: ${consoleErrors.length}`,
      `- Page errors: ${pageErrors.length}`,
      `- Unexpected HTTP errors: ${responseErrors.length}`,
      "",
      manifest.claim_boundary,
    ].join("\n"),
    "utf8",
  );
  console.log(JSON.stringify({ passed: manifest.passed, checks: checks.length, outputRoot }));
} catch (error) {
  const failure = {
    validation_version: "vitalssight.browser-product-validation.v3",
    passed: false,
    git_commit: commit,
    error: String(error?.stack || error),
    checks,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    response_errors: responseErrors,
  };
  await fs.writeFile(path.join(outputRoot, "browser_validation_manifest.json"), JSON.stringify(failure, null, 2), "utf8");
  await saveState(page, "FAILED_STATE").catch(() => {});
  throw error;
} finally {
  await context.close();
  if (mobileContext) await mobileContext.close();
  await browser.close();
}

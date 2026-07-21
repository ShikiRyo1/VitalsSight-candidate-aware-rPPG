import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const baseUrl = process.argv[2] || 'http://127.0.0.1:8788/';
const outputDir = path.resolve(process.argv[3] || 'output/project-page-validation');
await fs.mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const profiles = [
  { name: 'desktop', viewport: { width: 1440, height: 900 }, isMobile: false },
  { name: 'mobile', viewport: { width: 390, height: 844 }, isMobile: true },
];
const report = { baseUrl, status: 'PASS', profiles: [] };

try {
  for (const profile of profiles) {
    const context = await browser.newContext({
      viewport: profile.viewport,
      deviceScaleFactor: 1,
      isMobile: profile.isMobile,
      hasTouch: profile.isMobile,
    });
    const page = await context.newPage();
    const consoleErrors = [];
    const pageErrors = [];
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => pageErrors.push(error.message));

    const response = await page.goto(baseUrl, { waitUntil: 'networkidle' });
    if (!response || !response.ok()) {
      throw new Error(`${profile.name}: project page returned ${response?.status() ?? 'no response'}`);
    }

    await page.locator('h1').waitFor();
    const title = await page.locator('h1').innerText();
    if (title.trim() !== 'VitalsSight') throw new Error(`${profile.name}: unexpected h1 ${title}`);

    const imageCount = await page.locator('img').count();
    for (let index = 0; index < imageCount; index += 1) {
      await page.locator('img').nth(index).scrollIntoViewIfNeeded();
    }
    await page.waitForFunction(() => Array.from(document.images).every((image) => image.complete));

    const imageAudit = await page.locator('img').evaluateAll((images) => images.map((image) => ({
      src: image.getAttribute('src'),
      complete: image.complete,
      naturalWidth: image.naturalWidth,
      naturalHeight: image.naturalHeight,
    })));
    const brokenImages = imageAudit.filter((image) => !image.complete || image.naturalWidth === 0);
    if (brokenImages.length) throw new Error(`${profile.name}: broken images ${JSON.stringify(brokenImages)}`);

    const internalTargets = await page.locator('a[href^="#"]').evaluateAll((links) => links.map((link) => link.getAttribute('href')));
    const missingTargets = await page.evaluate((targets) => targets.filter((target) => target !== '#' && !document.querySelector(target)), internalTargets);
    if (missingTargets.length) throw new Error(`${profile.name}: missing hash targets ${missingTargets.join(', ')}`);

    const layout = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      scrollHeight: document.documentElement.scrollHeight,
      clientHeight: document.documentElement.clientHeight,
    }));
    if (layout.scrollWidth > layout.clientWidth + 1) {
      throw new Error(`${profile.name}: horizontal overflow ${layout.scrollWidth} > ${layout.clientWidth}`);
    }

    if (profile.isMobile) {
      const menuButton = page.locator('.menu-button');
      await menuButton.click();
      if ((await menuButton.getAttribute('aria-expanded')) !== 'true') throw new Error('mobile: menu did not open');
      await page.locator('#site-menu a[href="#method"]').click();
      if ((await menuButton.getAttribute('aria-expanded')) !== 'false') throw new Error('mobile: menu did not close after navigation');
    }

    await page.locator('.copy-button').scrollIntoViewIfNeeded();
    await page.locator('.copy-button').click();
    await page.waitForFunction(() => document.querySelector('.copy-button')?.textContent.trim() !== 'Copy URL');
    const copyLabel = (await page.locator('.copy-button').innerText()).trim();
    if (!['Copied', 'Selected'].includes(copyLabel)) throw new Error(`${profile.name}: copy control returned ${copyLabel}`);

    await page.screenshot({ path: path.join(outputDir, `${profile.name}-full.png`), fullPage: true });
    await page.locator('.hero').screenshot({ path: path.join(outputDir, `${profile.name}-hero.png`) });

    if (consoleErrors.length || pageErrors.length) {
      throw new Error(`${profile.name}: browser errors ${JSON.stringify({ consoleErrors, pageErrors })}`);
    }

    report.profiles.push({
      name: profile.name,
      viewport: profile.viewport,
      title,
      imageAudit,
      internalTargetCount: internalTargets.length,
      layout,
      copyControl: copyLabel,
      consoleErrors,
      pageErrors,
      status: 'PASS',
    });
    await context.close();
  }
} catch (error) {
  report.status = 'FAIL';
  report.error = error.stack || String(error);
  throw error;
} finally {
  await browser.close();
  await fs.writeFile(path.join(outputDir, 'project-page-validation.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
}

console.log(JSON.stringify({ status: report.status, outputDir, profiles: report.profiles.map((profile) => profile.name) }));

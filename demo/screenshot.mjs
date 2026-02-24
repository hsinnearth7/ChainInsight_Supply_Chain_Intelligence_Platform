import { chromium } from 'playwright';
import { existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEMO_DIR = __dirname;
const BASE = 'http://localhost:8000';

// Ensure demo dir exists
if (!existsSync(DEMO_DIR)) mkdirSync(DEMO_DIR, { recursive: true });

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function shot(page, name, waitMs = 1500) {
  await sleep(waitMs);
  const path = join(DEMO_DIR, `${name}.png`);
  await page.screenshot({ path, fullPage: true });
  console.log(`  ✅ ${name}.png`);
}

(async () => {
  console.log('🚀 Launching browser...');
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();

  // ──────────────────────────────────────
  // 1. Dashboard (EN)
  // ──────────────────────────────────────
  console.log('\n📊 Dashboard...');
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await sleep(2000);
  await shot(page, '01_dashboard_en');

  // 2. Dashboard (ZH)
  console.log('🌐 Switching to 中文...');
  await page.click('button:text("中文")');
  await shot(page, '02_dashboard_zh');

  // 3. Dashboard (JA)
  console.log('🌐 Switching to 日本語...');
  await page.click('button:text("日本語")');
  await shot(page, '03_dashboard_ja');

  // Switch back to EN for remaining screenshots
  await page.click('button:text("EN")');
  await sleep(500);

  // 4. Dark mode
  console.log('🌙 Dark mode...');
  await page.click('button:has-text("Dark")');
  await shot(page, '04_dashboard_dark');
  // Switch back to light
  await page.click('button:has-text("Light")');
  await sleep(500);

  // ──────────────────────────────────────
  // 5. Upload page (before upload)
  // ──────────────────────────────────────
  console.log('\n📤 Upload page...');
  await page.click('a[href="/upload"]');
  await shot(page, '05_upload_page');

  // 6. Trigger pipeline
  console.log('🔄 Triggering pipeline...');
  await page.click('button:has-text("dirty_inventory")');
  await sleep(3000);
  await shot(page, '06_pipeline_running');

  // 7. Wait for pipeline completion (up to 10 minutes)
  console.log('⏳ Waiting for pipeline to complete (up to 10 min)...');
  let completed = false;
  for (let i = 0; i < 120; i++) {
    const text = await page.textContent('body');
    if (text.includes('Pipeline complete') || text.includes('管線已完成') || text.includes('パイプライン完了')) {
      completed = true;
      break;
    }
    // Take a mid-run screenshot at ~30s and ~2min
    if (i === 6) await shot(page, '07_pipeline_mid_progress', 0);
    await sleep(5000);
  }

  if (completed) {
    console.log('✅ Pipeline completed!');
    await shot(page, '08_pipeline_completed');
  } else {
    console.log('⚠️ Pipeline did not complete in 10 min, taking screenshot anyway');
    await shot(page, '08_pipeline_timeout');
  }

  // ──────────────────────────────────────
  // 8. Stats page
  // ──────────────────────────────────────
  console.log('\n📈 Stats page...');
  await page.click('a[href="/stats"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '09_stats_interactive', 2000);

  // Stats PNG tab
  await page.click('button:has-text("PNG")');
  await shot(page, '10_stats_png', 3000);

  // Stats Raw Data tab
  await page.click('button:has-text("Raw Data")');
  await shot(page, '11_stats_rawdata', 2000);

  // ──────────────────────────────────────
  // 9. Supply Chain page
  // ──────────────────────────────────────
  console.log('\n🔗 Supply Chain page...');
  await page.click('a[href="/supply-chain"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '12_supply_chain_eoq', 2000);

  // Run Monte Carlo
  await page.click('button:has-text("Run Simulation")');
  await shot(page, '13_supply_chain_monte_carlo', 1500);

  // ──────────────────────────────────────
  // 10. ML page
  // ──────────────────────────────────────
  console.log('\n🤖 ML page...');
  await page.click('a[href="/ml"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '14_ml_page', 3000);

  // ──────────────────────────────────────
  // 11. RL page
  // ──────────────────────────────────────
  console.log('\n🎯 RL page...');
  await page.click('a[href="/rl"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '15_rl_page', 3000);

  // ──────────────────────────────────────
  // 12. History page
  // ──────────────────────────────────────
  console.log('\n📋 History page...');
  await page.click('a[href="/history"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '16_history_page', 2000);

  // ──────────────────────────────────────
  // 13. i18n demo - ZH full cycle
  // ──────────────────────────────────────
  console.log('\n🌐 i18n demo (中文 across pages)...');
  await page.click('button:text("中文")');
  await page.click('a[href="/supply-chain"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '17_supply_chain_zh', 2000);

  await page.click('a[href="/rl"]');
  await page.waitForLoadState('networkidle');
  await shot(page, '18_rl_zh', 2000);

  console.log('\n🎉 All screenshots saved to demo/ folder!');
  console.log(`   Total: ${18} screenshots`);

  await browser.close();
})();

#!/usr/bin/env node
const data = require("../coverage/coverage-summary.json");
const files = [];

for (const [path, stats] of Object.entries(data)) {
  if (path === "total") continue;
  const shortPath = path.replace("/Users/armand/Development/RingRift/", "");
  const branches = stats.branches;
  const uncovered = branches.total - branches.covered;
  if (branches.total >= 10) {  // Only files with significant branches
    files.push({
      path: shortPath,
      total: branches.total,
      covered: branches.covered,
      uncovered: uncovered,
      pct: branches.pct
    });
  }
}

// Sort by uncovered branches (most impact)
files.sort((a, b) => b.uncovered - a.uncovered);

console.log("TOP 25 FILES BY UNCOVERED BRANCHES (highest impact):\n");
console.log("Uncov | Total | Pct%  | Path");
console.log("------+-------+-------+------------------");
files.slice(0, 25).forEach(f => {
  console.log(`${String(f.uncovered).padStart(5)} | ${String(f.total).padStart(5)} | ${String(f.pct.toFixed(1)).padStart(5)} | ${f.path}`);
});

console.log("\n\nSummary by module:");
const byModule = {};
files.forEach(f => {
  let module = f.path.split("/").slice(0, 3).join("/");
  if (!byModule[module]) byModule[module] = { total: 0, covered: 0, uncovered: 0 };
  byModule[module].total += f.total;
  byModule[module].covered += f.covered;
  byModule[module].uncovered += f.uncovered;
});

const sorted = Object.entries(byModule).sort((a,b) => b[1].uncovered - a[1].uncovered);
console.log("\nUncov | Total | Pct%  | Module");
console.log("------+-------+-------+------------------");
sorted.forEach(([mod, s]) => {
  const pct = s.total > 0 ? (s.covered / s.total * 100).toFixed(1) : 0;
  console.log(`${String(s.uncovered).padStart(5)} | ${String(s.total).padStart(5)} | ${String(pct).padStart(5)} | ${mod}`);
});

// Calculate how much improvement we need
const total = data.total.branches;
const needed = Math.ceil(total.total * 0.70) - total.covered;
console.log(`\n\n=== TO REACH 70% BRANCH COVERAGE ===`);
console.log(`Current: ${total.covered}/${total.total} (${total.pct.toFixed(1)}%)`);
console.log(`Need to cover: ${needed} more branches`);
console.log(`Top 25 files have: ${files.slice(0, 25).reduce((sum, f) => sum + f.uncovered, 0)} uncovered branches`);

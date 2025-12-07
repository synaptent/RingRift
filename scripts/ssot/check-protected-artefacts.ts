#!/usr/bin/env ts-node
/**
 * Protected Artefacts Pre-commit Warning Script
 *
 * This script is meant to be run as part of a pre-commit hook to warn
 * developers when they modify protected artefacts that require validation.
 *
 * Usage:
 *   npx ts-node scripts/ssot/check-protected-artefacts.ts [--staged] [--verbose]
 *
 * Options:
 *   --staged   Check only git-staged files (for pre-commit hooks)
 *   --verbose  Show detailed information about each category
 *
 * Exit codes:
 *   0 - No protected files modified, or warning issued (non-blocking)
 *   1 - Error running the check
 *
 * This script intentionally does NOT block commits; it only warns.
 * Full validation happens in CI via the ssot-check job.
 */

import { execSync } from 'child_process';

import { getAffectedCategories, type ProtectedCategory } from './protected-artefacts.config';

interface CheckOptions {
  staged: boolean;
  verbose: boolean;
}

function parseArgs(): CheckOptions {
  const args = process.argv.slice(2);
  return {
    staged: args.includes('--staged'),
    verbose: args.includes('--verbose'),
  };
}

/**
 * Get the list of files modified according to git.
 */
function getModifiedFiles(staged: boolean): string[] {
  try {
    const command = staged
      ? 'git diff --cached --name-only --diff-filter=ACMR'
      : 'git diff --name-only --diff-filter=ACMR HEAD';

    const output = execSync(command, { encoding: 'utf8' });
    return output
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
  } catch {
    // Not in a git repo or git not available
    return [];
  }
}

/**
 * Format validation commands for display.
 */
function formatValidationCommands(commands: string[] | undefined): string {
  if (!commands || commands.length === 0) {
    return '  Run: npm run ssot-check';
  }
  return commands.map((cmd) => `  Run: ${cmd}`).join('\n');
}

/**
 * Print a warning banner for a protected category.
 */
function printCategoryWarning(category: ProtectedCategory, matchedFiles: string[]): void {
  /* eslint-disable no-console */
  console.log('');
  console.log(`⚠️  PROTECTED ARTEFACT: ${category.name.toUpperCase()}`);
  console.log(`   Level: ${category.level}`);
  console.log(`   ${category.description}`);
  console.log('');
  console.log('   Modified files:');
  for (const file of matchedFiles.slice(0, 5)) {
    console.log(`     - ${file}`);
  }
  if (matchedFiles.length > 5) {
    console.log(`     ... and ${matchedFiles.length - 5} more`);
  }
  console.log('');
  console.log('   Validation requirement:');
  console.log(`   ${category.validationRequirement}`);
  console.log('');
  console.log('   Before merging:');
  console.log(formatValidationCommands(category.validationCommands));
  /* eslint-enable no-console */
}

/**
 * Main check function.
 */
function main(): void {
  const options = parseArgs();
  const modifiedFiles = getModifiedFiles(options.staged);

  if (modifiedFiles.length === 0) {
    if (options.verbose) {
      // eslint-disable-next-line no-console
      console.log('No modified files to check.');
    }
    return;
  }

  // Find which categories are affected
  const affectedCategories = getAffectedCategories(modifiedFiles);

  if (affectedCategories.size === 0) {
    if (options.verbose) {
      // eslint-disable-next-line no-console
      console.log(`Checked ${modifiedFiles.length} files. No protected artefacts modified.`);
    }
    return;
  }

  // Collect affected files per category for detailed reporting
  const categorizedFiles = new Map<string, string[]>();

  for (const file of modifiedFiles) {
    for (const [name, category] of affectedCategories) {
      for (const pattern of category.patterns) {
        const regexPattern = pattern
          .replace(/\./g, '\\.')
          .replace(/\*\*/g, '{{GLOBSTAR}}')
          .replace(/\*/g, '[^/]*')
          .replace(/\{\{GLOBSTAR\}\}/g, '.*');

        const regex = new RegExp(`^${regexPattern}$`);
        if (regex.test(file)) {
          if (!categorizedFiles.has(name)) {
            categorizedFiles.set(name, []);
          }
          categorizedFiles.get(name)!.push(file);
          break;
        }
      }
    }
  }

  // eslint-disable-next-line no-console
  console.log('');
  // eslint-disable-next-line no-console
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  // eslint-disable-next-line no-console
  console.log('║         PROTECTED ARTEFACT MODIFICATION WARNING                  ║');
  // eslint-disable-next-line no-console
  console.log('╚══════════════════════════════════════════════════════════════════╝');

  // Print warnings for each affected category
  const highPriority: Array<[string, ProtectedCategory]> = [];
  const mediumPriority: Array<[string, ProtectedCategory]> = [];

  for (const [name, category] of affectedCategories) {
    if (category.level === 'HIGH') {
      highPriority.push([name, category]);
    } else {
      mediumPriority.push([name, category]);
    }
  }

  // Print HIGH priority first
  for (const [name, category] of highPriority) {
    printCategoryWarning(category, categorizedFiles.get(name) ?? []);
  }

  // Then MEDIUM
  for (const [name, category] of mediumPriority) {
    printCategoryWarning(category, categorizedFiles.get(name) ?? []);
  }

  // Summary
  /* eslint-disable no-console */
  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  console.log('║                           SUMMARY                                 ║');
  console.log('╚══════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`  Categories affected: ${affectedCategories.size}`);
  console.log(`    HIGH protection: ${highPriority.length}`);
  console.log(`    MEDIUM protection: ${mediumPriority.length}`);
  console.log('');
  console.log('  This is a warning only. Commit will proceed.');
  console.log('  CI will run full validation on push.');
  console.log('');
  console.log('  To run validation locally:');
  console.log('    npm run ssot-check');
  console.log('');
  /* eslint-enable no-console */
}

try {
  main();
} catch (error) {
  console.error('Error checking protected artefacts:', error);
  process.exitCode = 1;
}

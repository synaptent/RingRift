import fs from 'fs';
import path from 'path';

function readAllTsFiles(dir: string, ignore: (p: string) => boolean): string {
  let contents = '';
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (ignore(fullPath)) {
      continue;
    }
    if (entry.isDirectory()) {
      contents += readAllTsFiles(fullPath, ignore);
    } else if (entry.isFile() && fullPath.endsWith('.ts')) {
      contents += fs.readFileSync(fullPath, 'utf8');
      contents += '\n';
    }
  }
  return contents;
}

describe('No randomness in core TS rules engine', () => {
  it('shared engine contains no Math.random calls (excluding AI helpers and test utilities)', () => {
    const engineDir = path.join(__dirname, '..', '..', 'src', 'shared', 'engine');
    const text = readAllTsFiles(
      engineDir,
      (p) =>
        p.endsWith(`${path.sep}localAIMoveSelection.ts`) ||
        p.includes(`${path.sep}contracts${path.sep}`) // Exclude test utilities
    );
    expect(text).not.toMatch(/Math\.random/);
  });

  it('server game core (non-AI) contains no Math.random calls', () => {
    const gameDir = path.join(__dirname, '..', '..', 'src', 'server', 'game');
    const text = readAllTsFiles(gameDir, (p) => p.includes(`${path.sep}ai${path.sep}`));
    expect(text).not.toMatch(/Math\.random/);
  });
});

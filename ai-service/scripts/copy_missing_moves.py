#!/usr/bin/env python3
"""Copy missing move data from OWC imports to canonical databases.

December 29, 2025: Quick fix script to copy moves for games that were
consolidated but had their moves missed due to column mismatch bug.
"""

import sqlite3
from pathlib import Path


def main():
    # Source databases
    sources = [
        'data/games/owc_imports/selfplay_repository_consolidated_archives_synced_20251213_182629_vast_2x5090_selfplay.db',
        'data/games/owc_imports/selfplay_repository_consolidated_archives_synced_20251213_172954_vast-5090-quad_selfplay.db',
        'data/games/owc_imports/selfplay_repository_consolidated_archives_synced_20251213_182629_lambda-h100_selfplay.db',
        'data/games/owc_imports/training_data_coordinator_backup_sq19_4p_selfplay.db',
        'data/games/owc_imports/training_data_coordinator_backup_sq19_2p_selfplay.db',
    ]

    configs = [
        ('hexagonal', 4, 'canonical_hexagonal_4p.db'),
        ('hexagonal', 3, 'canonical_hexagonal_3p.db'),
        ('square19', 4, 'canonical_square19_4p.db'),
        ('square19', 3, 'canonical_square19_3p.db'),
        ('square19', 2, 'canonical_square19_2p.db'),
    ]

    for board_type, num_players, target_name in configs:
        target_path = Path(f'data/games/{target_name}')
        if not target_path.exists():
            continue

        target_conn = sqlite3.connect(str(target_path))

        # Get games in target without moves
        cursor = target_conn.execute('''
            SELECT g.game_id FROM games g
            LEFT JOIN game_moves m ON g.game_id = m.game_id
            WHERE m.game_id IS NULL
            AND g.board_type = ? AND g.num_players = ?
        ''', (board_type, num_players))
        missing_games = set(row[0] for row in cursor.fetchall())

        if not missing_games:
            print(f'{board_type}_{num_players}p: All games have moves')
            target_conn.close()
            continue

        print(f'{board_type}_{num_players}p: {len(missing_games)} games missing moves')

        # Copy moves from sources
        moves_copied = 0
        for source_path in sources:
            if not Path(source_path).exists():
                continue

            source_conn = sqlite3.connect(source_path)
            source_conn.row_factory = sqlite3.Row

            for game_id in list(missing_games):
                try:
                    # Get moves for this game
                    cursor = source_conn.execute('SELECT * FROM game_moves WHERE game_id = ?', (game_id,))
                    rows = cursor.fetchall()
                except sqlite3.DatabaseError as e:
                    print(f'    Skipping corrupt database: {source_path}')
                    break  # Skip this source

                if not rows:
                    continue

                # Get column names
                columns = [desc[0] for desc in cursor.description]

                # Get target columns
                target_cursor = target_conn.execute('PRAGMA table_info(game_moves)')
                target_cols = {row[1] for row in target_cursor.fetchall()}

                # Filter to common columns
                common_cols = [c for c in columns if c in target_cols]
                col_indices = [columns.index(c) for c in common_cols]

                # Insert moves
                filtered_rows = [tuple(list(row)[i] for i in col_indices) for row in rows]
                placeholders = ','.join(['?'] * len(common_cols))
                cols_str = ','.join(common_cols)
                target_conn.executemany(
                    f'INSERT OR IGNORE INTO game_moves ({cols_str}) VALUES ({placeholders})',
                    filtered_rows
                )

                missing_games.discard(game_id)
                moves_copied += len(rows)

            source_conn.close()

        target_conn.commit()
        target_conn.close()
        print(f'  Copied {moves_copied} moves, {len(missing_games)} still missing')


if __name__ == '__main__':
    main()

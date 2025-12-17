/**
 * Replay Viewer - Game replay logic and API integration
 */

const API_BASE = window.location.origin;

// State
let currentGameId = null;
let currentMove = 0;
let totalMoves = 0;
let moves = [];
let stateCache = {};
let boardType = 'square8';
let gamesOffset = 0;
const GAMES_LIMIT = 20;

// Board renderer instance
let renderer = null;

/**
 * Initialize the viewer
 */
function init() {
    renderer = new BoardRenderer('board-area');

    // Load initial games
    loadGames();

    // Set up event listeners
    document.getElementById('board-filter').addEventListener('change', () => {
        gamesOffset = 0;
        loadGames();
    });
    document.getElementById('winner-filter').addEventListener('change', () => {
        gamesOffset = 0;
        loadGames();
    });
    document.getElementById('game-search').addEventListener('input', debounce(() => {
        gamesOffset = 0;
        loadGames();
    }, 300));

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (!currentGameId) return;
        if (e.key === 'ArrowLeft') prevMove();
        if (e.key === 'ArrowRight') nextMove();
        if (e.key === 'Home') goToMove(0);
        if (e.key === 'End') goToMove(totalMoves);
    });

    // Check for game ID in URL
    const urlParams = new URLSearchParams(window.location.search);
    const gameId = urlParams.get('game') || window.location.pathname.split('/').pop();
    if (gameId && gameId !== 'replay') {
        loadGame(gameId);
    }
}

/**
 * Load games list
 */
async function loadGames() {
    const boardFilter = document.getElementById('board-filter').value;
    const winnerFilter = document.getElementById('winner-filter').value;
    const search = document.getElementById('game-search').value;

    const params = new URLSearchParams();
    params.set('limit', GAMES_LIMIT);
    params.set('offset', gamesOffset);
    if (boardFilter) params.set('board_type', boardFilter);
    if (winnerFilter) params.set('winner', winnerFilter);
    if (search) params.set('game_id', search);

    try {
        const resp = await fetch(`${API_BASE}/api/replay/games?${params}`);
        if (!resp.ok) throw new Error('Failed to fetch games');
        const data = await resp.json();

        renderGameList(data.games || []);
    } catch (e) {
        console.error('Error loading games:', e);
        document.getElementById('game-list').innerHTML =
            '<p style="color: #ff6b6b; padding: 10px;">Error loading games</p>';
    }
}

/**
 * Load more games
 */
function loadMoreGames() {
    gamesOffset += GAMES_LIMIT;
    loadGames();
}

/**
 * Render game list
 */
function renderGameList(games) {
    const container = document.getElementById('game-list');

    if (gamesOffset === 0) {
        container.innerHTML = '';
    }

    if (games.length === 0 && gamesOffset === 0) {
        container.innerHTML = '<p style="color: #888; padding: 10px;">No games found</p>';
        return;
    }

    games.forEach(game => {
        const item = document.createElement('div');
        item.className = 'game-item';
        if (game.gameId === currentGameId) {
            item.classList.add('selected');
        }
        item.dataset.gameId = game.gameId;
        item.onclick = () => loadGame(game.gameId);

        const winner = game.winner ? `P${game.winner} won` : 'Draw';
        const board = game.boardType || 'unknown';

        item.innerHTML = `
            <div class="game-id">${truncateId(game.gameId)}</div>
            <div class="game-info">${board} | ${game.totalMoves || '?'} moves | ${winner}</div>
        `;

        container.appendChild(item);
    });
}

/**
 * Load a specific game
 */
async function loadGame(gameId) {
    try {
        // Fetch game metadata
        const metaResp = await fetch(`${API_BASE}/api/replay/games/${gameId}`);
        if (!metaResp.ok) throw new Error('Game not found');
        const meta = await metaResp.json();

        currentGameId = gameId;
        boardType = meta.boardType || 'square8';
        totalMoves = meta.totalMoves || 0;
        currentMove = 0;
        stateCache = {};
        moves = [];

        // Update URL
        history.pushState({}, '', `/replay/${gameId}`);

        // Update game list selection
        document.querySelectorAll('.game-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.gameId === gameId);
        });

        // Update metadata display
        document.getElementById('meta-board').textContent = meta.boardType || '--';
        document.getElementById('meta-players').textContent = meta.numPlayers || '--';
        document.getElementById('meta-winner').textContent = meta.winner ? `Player ${meta.winner}` : 'Draw';
        document.getElementById('meta-moves').textContent = meta.totalMoves || '--';
        document.getElementById('meta-duration').textContent = formatDuration(meta.durationMs);

        // Fetch moves
        const movesResp = await fetch(`${API_BASE}/api/replay/games/${gameId}/moves?limit=2000`);
        if (movesResp.ok) {
            const movesData = await movesResp.json();
            moves = movesData.moves || [];
            renderMoveList();
        }

        // Show controls
        document.getElementById('controls').style.display = 'flex';

        // Update slider
        const slider = document.getElementById('move-slider');
        slider.max = totalMoves;
        slider.value = 0;

        // Load initial state
        await goToMove(0);

    } catch (e) {
        console.error('Error loading game:', e);
        alert('Error loading game: ' + e.message);
    }
}

/**
 * Go to specific move
 */
async function goToMove(moveNumber) {
    moveNumber = parseInt(moveNumber);
    if (moveNumber < 0) moveNumber = 0;
    if (moveNumber > totalMoves) moveNumber = totalMoves;

    // Fetch state if not cached
    if (!stateCache[moveNumber]) {
        try {
            const resp = await fetch(
                `${API_BASE}/api/replay/games/${currentGameId}/state?move_number=${moveNumber}`
            );
            if (resp.ok) {
                const data = await resp.json();
                stateCache[moveNumber] = data.gameState || data;
            }
        } catch (e) {
            console.error('Error fetching state:', e);
        }
    }

    currentMove = moveNumber;

    // Render board
    const state = stateCache[moveNumber];
    if (state) {
        // Get highlight positions from current move
        let highlightFrom = null;
        let highlightTo = null;

        if (moveNumber > 0 && moves[moveNumber - 1]) {
            const move = moves[moveNumber - 1];
            highlightFrom = move.from;
            highlightTo = move.to;
        }

        renderer.render(state, boardType, highlightFrom, highlightTo);
    }

    updateControls();
    highlightCurrentMove();
}

/**
 * Navigate to next move
 */
function nextMove() {
    if (currentMove < totalMoves) {
        goToMove(currentMove + 1);
    }
}

/**
 * Navigate to previous move
 */
function prevMove() {
    if (currentMove > 0) {
        goToMove(currentMove - 1);
    }
}

/**
 * Update control states
 */
function updateControls() {
    document.getElementById('btn-first').disabled = currentMove === 0;
    document.getElementById('btn-prev').disabled = currentMove === 0;
    document.getElementById('btn-next').disabled = currentMove >= totalMoves;
    document.getElementById('btn-last').disabled = currentMove >= totalMoves;

    document.getElementById('move-counter').textContent = `Move ${currentMove} / ${totalMoves}`;
    document.getElementById('move-slider').value = currentMove;
}

/**
 * Render move list
 */
function renderMoveList() {
    const container = document.getElementById('move-list');
    container.innerHTML = '';

    moves.forEach((move, i) => {
        const entry = document.createElement('div');
        entry.className = 'move-entry';
        entry.dataset.moveNum = i + 1;
        entry.onclick = () => goToMove(i + 1);

        const moveType = move.type || move.moveType || 'move';
        const player = move.player || '?';
        const desc = formatMove(move);

        entry.innerHTML = `
            <span class="move-num">${i + 1}.</span>
            P${player}: ${desc}
        `;

        container.appendChild(entry);
    });
}

/**
 * Highlight current move in list
 */
function highlightCurrentMove() {
    document.querySelectorAll('.move-entry').forEach(entry => {
        const num = parseInt(entry.dataset.moveNum);
        entry.classList.toggle('current', num === currentMove);
    });

    // Scroll to current move
    const current = document.querySelector('.move-entry.current');
    if (current) {
        current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
}

/**
 * Format move for display
 */
function formatMove(move) {
    const type = move.type || move.moveType || 'move';

    if (type === 'place_ring') {
        const to = move.to || move.position;
        return `Place ring at (${to?.x || '?'}, ${to?.y || '?'})`;
    }

    if (type === 'move_stack' || type === 'move') {
        const from = move.from;
        const to = move.to;
        return `Move (${from?.x || '?'},${from?.y || '?'}) -> (${to?.x || '?'},${to?.y || '?'})`;
    }

    if (type === 'overtaking_capture' || type === 'capture') {
        return `Capture at (${move.to?.x || '?'}, ${move.to?.y || '?'})`;
    }

    return type;
}

/**
 * Format duration in ms to readable string
 */
function formatDuration(ms) {
    if (!ms) return '--';
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs}s`;
}

/**
 * Truncate game ID for display
 */
function truncateId(id) {
    if (!id) return '--';
    if (id.length <= 20) return id;
    return id.substring(0, 8) + '...' + id.substring(id.length - 8);
}

/**
 * Debounce utility
 */
function debounce(fn, delay) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn.apply(this, args), delay);
    };
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

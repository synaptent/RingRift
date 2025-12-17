/**
 * Board Renderer - Renders RingRift boards (square and hex)
 */

const PLAYER_COLORS = {
    1: '#00d4ff',
    2: '#ff6b6b',
    3: '#4ecdc4',
    4: '#ffd93d'
};

const BOARD_CONFIGS = {
    square8: { type: 'square', size: 8 },
    square19: { type: 'square', size: 19 },
    hexagonal: { type: 'hex', radius: 4 },
    hex8: { type: 'hex', radius: 4 }
};

class BoardRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.boardType = null;
        this.size = 8;
    }

    /**
     * Render board based on type and state
     */
    render(state, boardType, highlightFrom = null, highlightTo = null) {
        this.boardType = boardType;
        const config = BOARD_CONFIGS[boardType] || BOARD_CONFIGS.square8;

        if (config.type === 'hex') {
            this.renderHexBoard(state, config.radius, highlightFrom, highlightTo);
        } else {
            this.renderSquareBoard(state, config.size, highlightFrom, highlightTo);
        }
    }

    /**
     * Render square board using CSS grid
     */
    renderSquareBoard(state, size, highlightFrom, highlightTo) {
        this.size = size;
        const cellSize = size <= 8 ? 50 : 30;

        const wrapper = document.createElement('div');
        wrapper.className = 'board-wrapper';

        const board = document.createElement('div');
        board.id = 'board';
        board.style.gridTemplateColumns = `repeat(${size}, ${cellSize}px)`;
        board.style.gridTemplateRows = `repeat(${size}, ${cellSize}px)`;

        // Get stacks and markers from state
        const stacks = state?.ringStacks || state?.board?.stacks || {};
        const markers = state?.markers || state?.board?.markers || {};

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;
                cell.dataset.x = x;
                cell.dataset.y = y;

                // Check for highlight
                if (highlightFrom && highlightFrom.x === x && highlightFrom.y === y) {
                    cell.classList.add('highlight-from');
                }
                if (highlightTo && highlightTo.x === x && highlightTo.y === y) {
                    cell.classList.add('highlight-to');
                }

                // Check for ring stack
                const key = `${x},${y}`;
                const stack = stacks[key];
                if (stack) {
                    const ring = document.createElement('div');
                    ring.className = `ring player-${stack.owner || stack.player}`;
                    ring.style.width = `${cellSize - 10}px`;
                    ring.style.height = `${cellSize - 10}px`;

                    // Show stack height
                    const height = stack.height || stack.count || 1;
                    if (height > 1) {
                        const stackLabel = document.createElement('span');
                        stackLabel.className = 'ring-stack';
                        stackLabel.textContent = height;
                        ring.appendChild(stackLabel);
                    }

                    cell.appendChild(ring);
                }

                // Check for marker
                const marker = markers[key];
                if (marker) {
                    const markerEl = document.createElement('div');
                    markerEl.className = `marker player-${marker.owner || marker.player}`;
                    cell.appendChild(markerEl);
                }

                board.appendChild(cell);
            }
        }

        wrapper.appendChild(board);
        this.container.innerHTML = '';
        this.container.appendChild(wrapper);
    }

    /**
     * Render hexagonal board using SVG
     */
    renderHexBoard(state, radius, highlightFrom, highlightTo) {
        const svgNS = 'http://www.w3.org/2000/svg';
        const hexSize = 30;
        const width = 500;
        const height = 500;

        const wrapper = document.createElement('div');
        wrapper.className = 'board-wrapper';

        const svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('id', 'hex-board');
        svg.setAttribute('viewBox', `-${width/2} -${height/2} ${width} ${height}`);
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);

        const stacks = state?.ringStacks || state?.board?.stacks || {};
        const markers = state?.markers || state?.board?.markers || {};

        // Iterate cube coordinates
        for (let q = -radius; q <= radius; q++) {
            for (let r = -radius; r <= radius; r++) {
                const s = -q - r;
                if (Math.abs(s) > radius) continue;

                const { x, y } = this.hexToPixel(q, r, hexSize);
                const key = `${q},${r},${s}`;
                const altKey = `${q},${r}`;

                // Create hex polygon
                const hex = document.createElementNS(svgNS, 'polygon');
                hex.setAttribute('points', this.hexPoints(x, y, hexSize));
                hex.setAttribute('class', 'hex');
                hex.dataset.q = q;
                hex.dataset.r = r;
                hex.dataset.s = s;

                // Highlight
                if (highlightFrom && highlightFrom.q === q && highlightFrom.r === r) {
                    hex.classList.add('highlight-from');
                }
                if (highlightTo && highlightTo.q === q && highlightTo.r === r) {
                    hex.classList.add('highlight-to');
                }

                svg.appendChild(hex);

                // Check for ring stack
                const stack = stacks[key] || stacks[altKey];
                if (stack) {
                    const circle = document.createElementNS(svgNS, 'circle');
                    circle.setAttribute('cx', x);
                    circle.setAttribute('cy', y);
                    circle.setAttribute('r', hexSize * 0.6);
                    circle.setAttribute('fill', PLAYER_COLORS[stack.owner || stack.player] || '#888');
                    svg.appendChild(circle);

                    // Stack height text
                    const height = stack.height || stack.count || 1;
                    if (height > 1) {
                        const text = document.createElementNS(svgNS, 'text');
                        text.setAttribute('x', x);
                        text.setAttribute('y', y + 5);
                        text.setAttribute('text-anchor', 'middle');
                        text.setAttribute('fill', '#1a1a2e');
                        text.setAttribute('font-size', '14');
                        text.setAttribute('font-weight', 'bold');
                        text.textContent = height;
                        svg.appendChild(text);
                    }
                }

                // Check for marker
                const marker = markers[key] || markers[altKey];
                if (marker) {
                    const markerCircle = document.createElementNS(svgNS, 'circle');
                    markerCircle.setAttribute('cx', x + hexSize * 0.4);
                    markerCircle.setAttribute('cy', y + hexSize * 0.4);
                    markerCircle.setAttribute('r', 5);
                    markerCircle.setAttribute('fill', PLAYER_COLORS[marker.owner || marker.player] || '#888');
                    svg.appendChild(markerCircle);
                }
            }
        }

        wrapper.appendChild(svg);
        this.container.innerHTML = '';
        this.container.appendChild(wrapper);
    }

    /**
     * Convert cube coordinates to pixel coordinates (pointy-top hex)
     */
    hexToPixel(q, r, size) {
        const x = size * (Math.sqrt(3) * q + Math.sqrt(3) / 2 * r);
        const y = size * (3 / 2 * r);
        return { x, y };
    }

    /**
     * Get hex polygon points
     */
    hexPoints(cx, cy, size) {
        const points = [];
        for (let i = 0; i < 6; i++) {
            const angle = Math.PI / 180 * (60 * i - 30);
            const x = cx + size * Math.cos(angle);
            const y = cy + size * Math.sin(angle);
            points.push(`${x},${y}`);
        }
        return points.join(' ');
    }

    /**
     * Clear the board
     */
    clear() {
        this.container.innerHTML = `
            <div class="empty-state">
                <h2>No Game Selected</h2>
                <p>Select a game from the browser to view the replay</p>
            </div>
        `;
    }
}

// Export for use
window.BoardRenderer = BoardRenderer;

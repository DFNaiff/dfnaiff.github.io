---
layout: default
title: Gravitational Golf
categories: Golf
permalink: /golf/
---

# Gravitational Golf
## A game by GPT-4 and Danilo Naiff

<canvas id="gameCanvas" width="800" height="400"></canvas>
<button id="reset-level" class="btn">Reset Level</button>
<button id="reset-game" class="btn">Reset Game</button>
<button id="instructions-button">Show Instructions</button>
<div id="instructions" style="display: none;">
    <h2>Instructions:</h2>
    <ol>
        <li>Click and drag the ball to set the launch direction and speed.</li>
        <li>Reach the target (black rectangle) to complete the level.</li>
        <li>The red (attractors) and blue (repellers) circles will hinder you.</li>
        <li>Click 'Reset Level' to restart the current level or 'Reset Game' to restart the entire game.</li>
    </ol>
</div>


<link rel="stylesheet" href="{{ '/assets/css/golf.css' | relative_url }}">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/matter-js/0.17.1/matter.min.js"></script>
<script src="{{ '/assets/js/golf.js' | relative_url }}"></script>

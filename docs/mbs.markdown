---
layout: default
title: MBS
categories: MBS
permalink: /mbs/
---


<link rel="stylesheet" href="{{ '/assets/css/mbs.css' | relative_url }}">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/p5@1.6.0/lib/p5.js"></script>
<script src="{{ '/assets/js/mbs.js' | relative_url }}"></script>

<h1>Quasielectrostatic Simulation</h1>
<div>
<label for="num-particles">Number of particles:</label>
<input type="number" id="num-particles" class="number-input" value="10" min="1">
<label for="canvas-width">Width:</label>
<input type="number" id="canvas-width" value="300" class="number-input" min="5" max="1200">
<label for="canvas-height">Height:</label>
<input type="number" id="canvas-height" value="300" class="number-input" min="5" max="1200">
</div>
<div>
<label for="k-constant">K:</label>
<input type="number" id="k-constant" value="100" class="number-input" step="0.1">
<label for="darwin-constant">D:</label>
<input type="number" id="darwin-constant" value="0" class="number-input" step="0.1">
<label for="magnetic-field">B:</label>
<input type="number" id="magnetic-field" value="0" class="number-input" step="0.1">
<label for="mass">M:</label>
<input type="number" id="mass" value="1.0" class="number-input" step="0.1">
</div>
<div>
<label for="time-step-size">Step Size:</label>
<input type="number" id="time-step-size" value="0.1" class="number-input" step="0.01">
<label for="trace-line-size">Trace Size:</label>
<input type="number" id="trace-line-size" class="number-input" value="100">
<label for="initial-speed">Initial Speed:</label>
<input type="number" id="initial-speed" class="number-input" value="1">
<div>
<button id="restart-button">Restart Simulation</button>
<button id="fullscreen-button">Fullscreen</button>
</div>
</div>
<div id="simulation-container"></div>
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
<input type="number" id="num-particles" value="10" min="1">
</div>
<div>
<label for="k-constant">k constant:</label>
<input type="number" id="k-constant" value="100" step="0.1">
<br>
<label for="darwin-constant">darwin constant:</label>
<input type="number" id="darwin-constant" value="0" step="0.1">
<br>
<label for="magnetic-field">Magnetic Field (B):</label>
<input type="number" id="magnetic-field" value="0" step="0.1">
<br>
<label for="mass">Mass:</label>
<input type="number" id="mass" value="1.0" step="0.1">
<label for="time-step-size">Time Step Size:</label>
<input type="number" id="time-step-size" value="0.1" step="0.01">
</div>
<div>
<label for="tracked-ball">Tracked Ball:</label>
<input type="number" id="tracked-ball" value="0">
<label for="trace-line-size">Trace Line Size:</label>
<input type="number" id="trace-line-size" value="100">
<div>
<button id="restart-button">Restart Simulation</button>
<button id="restart-tracked-ball-button">Update Tracked Ball</button>
</div>
</div>
<div id="simulation-container"></div>
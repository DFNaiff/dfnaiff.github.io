var A = 1.0, B = 1.0, C = 1.0, numParticles = 100, timeSpan = 10.0;
var particles = [], lines = [];
var scene, camera, renderer, gui;

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    camera.position.z = 5;

    gui = new dat.GUI();
    gui.add(this, 'A', -2.0, 2.0);
    gui.add(this, 'B', -2.0, 2.0);
    gui.add(this, 'C', -2.0, 2.0);
    gui.add(this, 'numParticles', 10, 1000).step(10);
    gui.add(this, 'timeSpan', 1.0, 20.0);

    for (var i = 0; i < numParticles; i++) {
        var material = new THREE.LineBasicMaterial({ color: Math.random() * 0xffffff });
        var geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]);
        var line = new THREE.Line(geometry, material);
        scene.add(line);
        lines.push(line);
    }

    window.requestAnimationFrame(update);
}

function abcFlow(xyz) {
  var x = xyz[0], y = xyz[1], z = xyz[2];
  var dxdt = A * Math.sin(z) + C * Math.cos(y);
  var dydt = B * Math.sin(x) + A * Math.cos(z);
  var dzdt = C * Math.sin(y) + B * Math.cos(x);

  // Check for NaN values
  if (isNaN(dxdt) || isNaN(dydt) || isNaN(dzdt)) {
      console.error("abcFlow returns NaN: ", dxdt, dydt, dzdt);
      return [0, 0, 0];  // Return a default value to avoid errors
  }
  
  return [dxdt, dydt, dzdt];
}


function update(t) {
  particles = numeric.linspace([0, 0, 0], [1, 1, 1], numParticles);
  for (var i = 0; i < numParticles; i++) {
      var particle = particles[i];
      var solution = numeric.dopri(0, timeSpan, particle, abcFlow, 1e-6, 2000);
      particle = solution.at(timeSpan);
      break
      // Check for NaN values
      if (isNaN(particle[0]) || isNaN(particle[1]) || isNaN(particle[2])) {
          console.error("Particle position is NaN: ", particle);
          continue;  // Skip this particle
      }
      
      var points = lines[i].geometry.attributes.position.array;
  

      for (var j = 0; j < points.length / 3; j++) {
          points[j * 3 + 0] = particle[0];
          points[j * 3 + 1] = particle[1];
          points[j * 3 + 2] = particle[2];
      }
      lines[i].geometry.attributes.position.needsUpdate = true;
  }
  renderer.render(scene, camera);
  window.requestAnimationFrame(update);
}

document.getElementById('startButton').addEventListener('click', init);

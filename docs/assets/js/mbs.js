// Moving charge class
class MovingCharge {
  constructor(position, velocity, charge, mass) {
    this.position = position;
    this.velocity = velocity;
    this.charge = charge;
    this.mass = mass;
  }

  update(dt) {
    this.position.add(p5.Vector.mult(this.velocity, dt));
  }

}

// Charged box class
class ChargedBox {
  constructor(width, height) {
    this.width = width;
    this.height = height;
  }

  isInside(position) {
    return position.x > 0 && position.x < this.width && position.y > 0 && position.y < this.height;
  }
}

// Constants
let k = 100; // Electrostatic constant (set to 1 for simplicity)
let darwinConstant = 0.0;
let B = 0;
let maxEnergyDataPoints = 400;

// Compute electrostatic force between two charges
function computeForce(charge1, charge2) {
  const eps = 1e-5; // Small constant to avoid division by zero

  let r = p5.Vector.sub(charge2.position, charge1.position);
  let distanceSq = r.magSq() + eps*eps;
  let forceMagnitude = -k * charge1.charge * charge2.charge / distanceSq;
  
  let v1v2 = p5.Vector.dot(charge1.velocity, charge2.velocity);
  let darwinForceMagnitude = 2 * darwinConstant * v1v2 * charge1.charge * charge2.charge / (distanceSq * distanceSq);
//   let darwinForce = r.normalize().mult(darwinForceMagnitude);
    let force = r.normalize().mult(forceMagnitude + darwinForceMagnitude)
//  let totalForce = force.add(darwinForce)
  return force;
}

function computeMagneticForce(charge, magneticField) {
  let force = p5.Vector.cross(charge.velocity, createVector(0, 0, magneticField));
  force.mult(charge.charge);
  return force;
}

function computeForceField(x, y, charges) {
  let fictitiousCharge = new MovingCharge(createVector(x, y), createVector(0, 0), 1, 1);
  let totalForce = createVector(0, 0);

  for (let charge of charges) {
    let force = computeForce(fictitiousCharge, charge);
    totalForce.add(force);
  }

  let magneticForce = computeMagneticForce(fictitiousCharge, B);
  totalForce.add(magneticForce);

  return totalForce;
}

function updateSimulation(charges, box, dt) {
    // Compute the initial accelerations
    let accelerations = [];
    for (let i = 0; i < charges.length; i++) {
      let totalForce = createVector(0, 0);
  
      for (let j = 0; j < charges.length; j++) {
        if (i !== j) {
          let force = computeForce(charges[i], charges[j]);
          totalForce.add(force);
        }
      }
  
      let magneticForce = computeMagneticForce(charges[i], B);
      totalForce.add(magneticForce);
  
      let acceleration = p5.Vector.div(totalForce, charges[i].mass);
      accelerations.push(acceleration);
    }
  
    // Update positions and handle reflections off the walls
    for (let i = 0; i < charges.length; i++) {
      let charge = charges[i];
      let oldPosition = charge.position.copy();
  
      // Update position
      charge.position.add(p5.Vector.mult(charge.velocity, dt)).add(p5.Vector.mult(accelerations[i], 0.5 * dt * dt));
  
      // Reflect velocity if outside the box and clamp the position
      if (charge.position.x < 0 || charge.position.x > box.width) {
        let deltaX = charge.position.x < 0 ? -charge.position.x : box.width - charge.position.x;
        charge.position.x += 2 * deltaX;
        charge.velocity.x *= -1;
      }
      if (charge.position.y < 0 || charge.position.y > box.height) {
        let deltaY = charge.position.y < 0 ? -charge.position.y : box.height - charge.position.y;
        charge.position.y += 2 * deltaY;
        charge.velocity.y *= -1;
      }
  
      // Compute new acceleration
      let newTotalForce = createVector(0, 0);
      for (let j = 0; j < charges.length; j++) {
        if (i !== j) {
          let force = computeForce(charge, charges[j]);
          newTotalForce.add(force);
        }
      }
      let newAcceleration = p5.Vector.div(newTotalForce, charge.mass);
  
      // Update the velocity using the average of old and new accelerations
      charge.velocity.add(p5.Vector.add(accelerations[i], newAcceleration).mult(0.5 * dt));
    }
  }
    
function computeTotalEnergy(charges) {
  let kineticEnergy = 0;
  let potentialEnergy = 0;

  for (let i = 0; i < charges.length; i++) {
    let charge1 = charges[i];
    kineticEnergy += 0.5 * charge1.mass * charge1.velocity.magSq();

    for (let j = i + 1; j < charges.length; j++) {
      let charge2 = charges[j];
      let distance = charge1.position.dist(charge2.position);
      potentialEnergy += charge1.mass * k * charge1.charge * charge2.charge / distance;
    }
  }

  return kineticEnergy + potentialEnergy;
}

// Global variables
let charges = [];
let box;
let dt = 0.1;

// User interface elements
let numParticlesInput;
let kConstantInput;
let darwinConstantInput;
let restartButton;
let traceLineSizeInput;
let massInput;
let timeStepSizeInput;
let magneticFieldInput;
let initialSpeedInput;
let canvasWidthInput;
let canvasHeightInput;
let fullscreenButton;
let showElectricFieldToggle;
let gridSpacingInput;

// Storing data
let totalEnergy = [];
let energyPlotHeight = 100;

// Tracing variables
let pastPositions = [];
let maxPastPositions = 100;

function restartSimulation() {
  // Update the number of particles, k constant, mass, time step size, and magnetic field
  const numParticles = parseInt(numParticlesInput.value());
  const newK = parseFloat(kConstantInput.value());
  const newD = parseFloat(darwinConstantInput.value());
  const newMass = parseFloat(massInput.value());
  const newDt = parseFloat(timeStepSizeInput.value());
  const newB = parseFloat(magneticFieldInput.value());
  const initialSpeed = parseFloat(initialSpeedInput.value());
  const newWidth = parseInt(canvasWidthInput.value());
  const newHeight = parseInt(canvasHeightInput.value());

  if (newWidth < 5 || newWidth > 1200 || newHeight < 5 || newHeight > 1200) {
    alert("Width and height values must be between 5 and 1200.");
    return;
  }

  if (!isNaN(newWidth) && !isNaN(newHeight)) {
    resizeCanvas(newWidth, newHeight + energyPlotHeight);
    box.width = width;
    box.height = height - energyPlotHeight;
  }

  if (!isNaN(numParticles) && !isNaN(newK) && !isNaN(newD) && !isNaN(newMass) && !isNaN(newDt) && !isNaN(newB) && !isNaN(initialSpeed)) {    k = newK;
    darwinConstant = newD;
    dt = newDt;
    B = newB;
    charges = [];

    // Initialize the moving charges with the new values
    for (let i = 0; i < numParticles; i++) {
      let position = createVector(random(width), random(height));
      
      // Randomize the velocity direction and apply the user-defined initial speed
      let angle = random(TWO_PI);
      let velocity = createVector(initialSpeed * cos(angle), initialSpeed * sin(angle));
      
      let charge = 10;
      charges.push(new MovingCharge(position, velocity, charge, newMass));
    }
    pastPositions = Array(charges.length).fill().map(() => []);
  }
}

function setup() {
  // Create the canvas
  const canvas = createCanvas(300, 300);
  canvas.parent('simulation-container');

  // Initialize the charged box
  box = new ChargedBox(width, height);

  // Initialize the moving charges and pastPositions
  for (let i = 0; i < 10; i++) {
    let position = createVector(random(width), random(height));
    let velocity = createVector(random(-1, 1), random(-1, 1));
    let charge = 10;
    charges.push(new MovingCharge(position, velocity, charge));
    pastPositions.push([]);
  }

  // Set canvas size to 400x400
  resizeCanvas(300, 300 + energyPlotHeight);

  // Get user interface elements
  numParticlesInput = select('#num-particles');
  kConstantInput = select('#k-constant');
  darwinConstantInput = select('#darwin-constant')
  restartButton = select('#restart-button');
  traceLineSizeInput = select('#trace-line-size');
  massInput = select('#mass');
  timeStepSizeInput = select('#time-step-size');
  magneticFieldInput = select('#magnetic-field');
  initialSpeedInput = select('#initial-speed');
  canvasWidthInput = select('#canvas-width');
  canvasHeightInput = select('#canvas-height');
  fullscreenButton = select('#fullscreen-button');
  showElectricFieldToggle = select('#show-electric-field');
  gridSpacingInput = select('#grid-spacing');

  // Add event listener to restart the simulation
  restartButton.mousePressed(restartSimulation);
  fullscreenButton.mousePressed(resizeCanvasAndToggleFullscreen);

}

function computeElectricField(point, charges) {
  let electricField = createVector(0, 0);
  for (let charge of charges) {
    let r = p5.Vector.sub(charge.position, point);
    let distanceSq = r.magSq();
    let fieldMagnitude = k * charge.charge / distanceSq;
    let field = r.normalize().mult(fieldMagnitude);
    electricField.add(field);
  }
  return electricField;
}

function draw() {
   background(0); // Set background to black
   push();
    translate((width - box.width) / 2, (height - box.height - energyPlotHeight) / 2);
    // Clear the canvas
    background(0);
    // Draw the charged box
    drawChargedBox(box);

    // Get the trace line size from the input field
    let traceLineSize = parseInt(traceLineSizeInput.value());
    if (isNaN(traceLineSize)) {
      traceLineSize = 100; // Default value
    }
  
    for (let p = 0; p < charges.length; p++) {
      for (let i = 0; i < pastPositions[p].length - 1; i++) {
        let alpha = map(i, 0, pastPositions[p].length - 1, 0, 255);
        stroke(0, 0, 255, alpha);
        line(pastPositions[p][i].x, pastPositions[p][i].y, pastPositions[p][i + 1].x, pastPositions[p][i + 1].y);
      }
    }
  
    // Update the simulation
    updateSimulation(charges, box, dt);
  
    // Store the past position for all particles
    for (let p = 0; p < charges.length; p++) {
      pastPositions[p].push(charges[p].position.copy());
      if (pastPositions[p].length > traceLineSize) {
        pastPositions[p].shift();
      }
    }
    
    // Draw the moving charges
    //for (let charge of charges) {
    //  drawMovingCharge(charge);
    //}
    if (showElectricFieldToggle.checked()) {
      let gridSpacing = parseInt(gridSpacingInput.value());
      if (isNaN(gridSpacing)) {
        gridSpacing = 20; // Default value
      }
      let B = parseFloat(magneticFieldInput.value());
      if (isNaN(B)) {
        B = 0; // Default value
      }
      drawElectricField(gridSpacing, B);
    }

    drawEnergyPlot(totalEnergy, energyPlotHeight);
    pop();
  }

function drawChargedBox(box) {
  stroke(0);
  fill(0);
  rect(0, 0, box.width, box.height);
}

function drawMovingCharge(charge) {
  stroke(255, 0, 0);
  strokeWeight(4); // Increase this value to make the red point larger
  point(charge.position.x, charge.position.y);
}

function drawElectricField(gridSpacing, B) {
  for (let x = 0; x <= width; x += gridSpacing) {
    for (let y = 0; y <= height; y += gridSpacing) {
      console.log(x, y)
      let point = createVector(x, y);
      let forceField = computeForceField(point.x, point.y, charges);
      let forceMagnitude = forceField.mag();
      // Cap the force magnitude to avoid infinities
      forceMagnitude = min(forceMagnitude, 1000);

      // Map the color from yellow (lowest magnitude) to red (highest magnitude)
      let forceColor = lerpColor(color(255, 0, 0, 10), color(255, 0, 0, 100), forceMagnitude / 10);
      stroke(forceColor);

      // Draw an arrow representing the force field
      push();
      translate(point.x, point.y);
      rotate(forceField.heading());
      line(0, 0, gridSpacing * 0.5, 0);
      line(gridSpacing * 0.5, 0, gridSpacing * 0.3, gridSpacing * 0.1);
      line(gridSpacing * 0.5, 0, gridSpacing * 0.3, -gridSpacing * 0.1);
      pop();
    }
  }
}

function drawEnergyPlot(totalEnergy, plotHeight) {
    // Draw the plot background
    fill(0);
    noStroke();
    rect(0, height - plotHeight, width, plotHeight);
  
    // Set up the plot style
    stroke(0, 0, 255);
    strokeWeight(1);
    noFill();
  
    // Scale the plot to fit in the plot area
    let maxEnergy = Math.max(...totalEnergy);
    let scaleY = plotHeight / maxEnergy;
  
    // Draw the energy plot
    beginShape();
    for (let i = 0; i < totalEnergy.length; i++) {
      vertex(width * (i / maxEnergyDataPoints), height - totalEnergy[i] * scaleY);
    }
    endShape();
  }

  function resizeCanvasAndToggleFullscreen() {
    let fs = fullscreen();
    fullscreen(!fs);
  }
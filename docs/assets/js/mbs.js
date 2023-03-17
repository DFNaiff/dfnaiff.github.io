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
  let r = p5.Vector.sub(charge2.position, charge1.position);
  let distanceSq = r.magSq();
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
let trackedBallInput;
let traceLineSizeInput;
let massInput;
let timeStepSizeInput;
let magneticFieldInput;

// Storing data
let totalEnergy = [];
let energyPlotHeight = 100;

// Tracing variables
let trackedBall = 0; // The index of the tracked ball
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

  if (!isNaN(numParticles) && !isNaN(newK) && !isNaN(newD) && !isNaN(newMass) && !isNaN(newDt) && !isNaN(newB)) {
    k = newK;
    d = newD;
    dt = newDt;
    B = newB;
    charges = [];

    // Initialize the moving charges with the new values
    for (let i = 0; i < numParticles; i++) {
      let position = createVector(random(width), random(height));
      let velocity = createVector(random(-1, 1), random(-1, 1));
      let charge = 10;
      charges.push(new MovingCharge(position, velocity, charge, newMass));
    }
  }
}

function setup() {
  // Create the canvas
  const canvas = createCanvas(300, 300);
  canvas.parent('simulation-container');

  // Initialize the charged box
  box = new ChargedBox(width, height);

  // Initialize the moving charges
  for (let i = 0; i < 10; i++) {
    let position = createVector(random(width), random(height));
    let velocity = createVector(random(-1, 1), random(-1, 1));
    let charge = 10;
    charges.push(new MovingCharge(position, velocity, charge));
  }
  // Set canvas size to 400x400
  resizeCanvas(300, 300 + energyPlotHeight);

  // Get user interface elements
  numParticlesInput = select('#num-particles');
  kConstantInput = select('#k-constant');
  darwinConstantInput = select('#darwin-constant')
  restartButton = select('#restart-button');
  traceLineSizeInput = select('#trace-line-size');
  trackedBallInput = select('#tracked-ball');
  restartTrackedBallButton = select('#restart-tracked-ball-button');
  massInput = select('#mass');
  timeStepSizeInput = select('#time-step-size');
  magneticFieldInput = select('#magnetic-field');

  // Add event listener to restart the simulation
  restartButton.mousePressed(restartSimulation);
  
  // Add event listener to update the tracked ball
  restartTrackedBallButton.mousePressed(() => {
    const newTrackedBall = parseInt(trackedBallInput.value());
    if (!isNaN(newTrackedBall) && newTrackedBall >= 0 && newTrackedBall < charges.length) {
      trackedBall = newTrackedBall;
      pastPositions = [];
    }
  });  
}

function draw() {
    // Clear the canvas
    background(240);
    // Draw the charged box
    drawChargedBox(box);

    // Get the trace line size from the input field
    let traceLineSize = parseInt(traceLineSizeInput.value());
    if (isNaN(traceLineSize)) {
      traceLineSize = 100; // Default value
    }
  
    // Draw the trace lines for the tracked ball
    for (let i = 0; i < pastPositions.length - 1; i++) {
      let alpha = map(i, 0, pastPositions.length - 1, 0, 255);
      stroke(0, 0, 255, alpha);
      line(pastPositions[i].x, pastPositions[i].y, pastPositions[i + 1].x, pastPositions[i + 1].y);
    }
  
    // Update the simulation
    updateSimulation(charges, box, dt);
  
    // Store the past position of the tracked ball
    pastPositions.push(charges[trackedBall].position.copy());
    if (pastPositions.length > traceLineSize) {
      pastPositions.shift();
    }
    
    // Draw the moving charges
    for (let charge of charges) {
      drawMovingCharge(charge);
    }
  
    drawEnergyPlot(totalEnergy, energyPlotHeight);
  }

function drawChargedBox(box) {
  stroke(0);
  fill(255);
  rect(0, 0, box.width, box.height);
}

function drawMovingCharge(charge) {
  stroke(0);
  fill(255, 0, 0);
  circle(charge.position.x, charge.position.y, 10);
}

function drawEnergyPlot(totalEnergy, plotHeight) {
    // Draw the plot background
    fill(255);
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
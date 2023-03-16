const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const { Engine, Render, World, Bodies, Events } = Matter;

const engine = Engine.create();
engine.gravity = { x: 0, y: 0 }

const ball = Bodies.circle(50, canvas.height / 2, 10, { restitution: 0.5, label: 'ball' });
const target = Bodies.rectangle(canvas.width - 50, canvas.height / 2, 20, 20, { isStatic: true, label: 'target' });

World.add(engine.world, [ball, target]);

const K = 200, ATTRACTOR_CHARGE = 1, REPELLER_CHARGE = -0.3;
const ATTRACTOR_AMPLITUDE = 100, REPELLER_AMPLITUDE = 0;
const ATTRACTOR_SPEED = -12.5, REPELLER_SPEED = 12.5;
const MAX_INITIAL_SPEED = 2;
let currentLevel = 0;
let deathCount = 0;
let levelCount = 1;

function getRandomPosition(min, max) {
    return Math.random() * (max - min) + min;
}

function setupLevel(level) {
    engine.world.bodies
        .filter(body => body.label === 'attractor')
        .forEach(attractor => World.remove(engine.world, attractor));

    const minDistance = 100;
    for (let i = 0; i < 2 * level + 1; i++) {
        let x, y;
        do {
            x = getRandomPosition(200, canvas.width - 200);
            y = getRandomPosition(100, canvas.height - 100);
        } while (Math.hypot(x - ball.position.x, y - ball.position.y) < minDistance || Math.hypot(x - target.position.x, y - target.position.y) < minDistance);

        const attractor = Bodies.circle(x, y, 10, { isStatic: true, label: 'attractor' });
        attractor.charge = i % 2 === 0 ? ATTRACTOR_CHARGE : REPELLER_CHARGE;
        World.add(engine.world, attractor);
    }
}

function nextLevel() {
    currentLevel++;
    levelCount++;
    setupLevel(currentLevel);
}

function updateAttractorRepellerPositions() {
    const t = engine.timing.timestamp * 0.001;

//    target.position.y = 200 + Math.sin(t * ATTRACTOR_SPEED) * ATTRACTOR_AMPLITUDE;
//    ball.circleRadius = Math.max(2, 10 - currentLevel);
//    target.bounds.max.x = target.bounds.min.x + Math.max(2, 20 - 2 * currentLevel);
//    target.bounds.max.y = target.bounds.min.y + Math.max(2, 20 - 2 * currentLevel);
}

function applyCoulombsLaw() {
    const chargedObjects = engine.world.bodies.filter(obj => obj.charge !== undefined);

    chargedObjects.forEach(obj => {
        const distanceX = ball.position.x - obj.position.x;
        const distanceY = ball.position.y - obj.position.y;
        const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);
        const ballCharge = -1.0
        const forceMagnitude = (K * ballCharge * obj.charge) / (distance * distance);
        const forceX = (forceMagnitude * distanceX) / distance;
        const forceY = (forceMagnitude * distanceY) / distance;

        Matter.Body.applyForce(ball, ball.position, { x: forceX, y: forceY });
    });
}

const TIME_STEP = 1 / 600, DAMPING = 0.9;

function applyLeapfrogIntegration(deltaTime) {
    const halfDeltaTime = deltaTime / 2;
    ball.velocity.x += (ball.force.x / ball.mass) * halfDeltaTime;
    ball.velocity.y += (ball.force.y / ball.mass) * halfDeltaTime;

    ball.position.x += ball.velocity.x * deltaTime;
    ball.position.y += ball.velocity.y * deltaTime;

    ball.velocity.x *= DAMPING;
    ball.velocity.y *= DAMPING;

    ball.velocity.x += (ball.force.x / ball.mass) * halfDeltaTime;
    ball.velocity.y += (ball.force.y / ball.mass) * halfDeltaTime;
}

let isPlaying = false, hasWon = false, hasLost = false;

Events.on(engine, 'collisionStart', (event) => {
    event.pairs.forEach((pair) => {
        if (pair.bodyA.label === 'ball' || pair.bodyB.label === 'ball') {
            const otherBody = pair.bodyA.label === 'ball' ? pair.bodyB : pair.bodyA;

            if (otherBody.label === 'target') {
                hasWon = true;
                isPlaying = false;
            }
        }
    });
});

let isMouseDown = false;
let mouseDownPosition = { x: 0, y: 0 };
let mouseCurrentPosition = { x: 0, y: 0};
let isStarted = false;

function capVelocity(launchVector) {
    launchX = launchVector.x;
    launchY = launchVector.y;
    launchNorm = Math.sqrt(launchX + launchY);
    if (launchNorm > MAX_INITIAL_SPEED){
        launchVector.x = launchVector.x*MAX_INITIAL_SPEED/launchNorm;
        launchVector.y = launchVector.x*MAX_INITIAL_SPEED/launchNorm;
    }
}

function handleStart(event) {
    event.preventDefault();
    isStarted = true;
    isMouseDown = true;
    const { clientX, clientY } = event.touches ? event.touches[0] : event;
    mouseDownPosition = { x: clientX, y: clientY };
}

canvas.addEventListener('mousedown', handleStart);
canvas.addEventListener('touchstart', handleStart);

function handleEnd(event) {
    event.preventDefault();
    if (isMouseDown) {
        const { clientX, clientY } = event.changedTouches ? event.changedTouches[0] : event;
        const mouseUpPosition = { x: clientX, y: clientY };
        const launchVector = {
            x: -(mouseDownPosition.x - mouseUpPosition.x) * 0.1,
            y: -(mouseDownPosition.y - mouseUpPosition.y) * 0.1,
        };
        capVelocity(launchVector);
        if (!isPlaying) {
            isPlaying = true;
            hasWon = false;
            hasLost = false;
        }
        Matter.Body.setVelocity(ball, launchVector);
        isMouseDown = false;
    }
}

canvas.addEventListener('mouseup', handleEnd);
canvas.addEventListener('touchend', handleEnd);

function handleMove(event) {
    event.preventDefault();
    const { clientX, clientY } = event.touches ? event.touches[0] : event;
    mouseCurrentPosition = { x: clientX, y: clientY };
}

canvas.addEventListener('mousemove', handleMove);
canvas.addEventListener('touchmove', handleMove);

function drawLaunchGuide(currentPosition) {
    const launchVector = {
        x: (mouseDownPosition.x - currentPosition.x) * 0.1,
        y: (mouseDownPosition.y - currentPosition.y) * 0.1,
    };
    capVelocity(launchVector);

    const guideEndPoint = {
        x: ball.position.x - launchVector.x * 10,
        y: ball.position.y - launchVector.y * 10,
    };

    ctx.beginPath();
    ctx.moveTo(ball.position.x, ball.position.y);
    ctx.lineTo(guideEndPoint.x, guideEndPoint.y);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    const arrowheadLength = -10;
    const arrowheadAngle = Math.atan2(launchVector.y, launchVector.x) + Math.PI / 6;
    const arrowheadEndPoint1 = {
        x: guideEndPoint.x - arrowheadLength * Math.cos(arrowheadAngle),
        y: guideEndPoint.y - arrowheadLength * Math.sin(arrowheadAngle),
    };
    const arrowheadEndPoint2 = {
        x: guideEndPoint.x - arrowheadLength * Math.cos(arrowheadAngle - Math.PI / 3),
        y: guideEndPoint.y - arrowheadLength * Math.sin(arrowheadAngle - Math.PI / 3),
    };

    ctx.beginPath();
    ctx.moveTo(guideEndPoint.x, guideEndPoint.y);
    ctx.lineTo(arrowheadEndPoint1.x, arrowheadEndPoint1.y);
    ctx.lineTo(arrowheadEndPoint2.x, arrowheadEndPoint2.y);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fill();
}

function resetGame() {
    if (hasLost) {
        deathCount++;
    }
    if (hasWon) {
        setupLevel(currentLevel);
    }
    Matter.Body.setPosition(ball, { x: 50, y: canvas.height / 2 });
    Matter.Body.setVelocity(ball, { x: 0, y: 0 });

    isStarted = false;
    isPlaying = false;
    hasWon = false;
    hasLost = false;

}

function gameLoop() {
    const deltaTime = TIME_STEP;

    if (isStarted) {
        Engine.update(engine, deltaTime * 1000);
        updateAttractorRepellerPositions();
    }

    if (isPlaying) {
        applyCoulombsLaw();
        applyLeapfrogIntegration(TIME_STEP);
    }

    if (hasWon) {
        console.log('You won the level!');
        nextLevel();
        resetGame();
    } else if (hasLost) {
        console.log('You lost the level!');
        resetGame();
    }

    if (ball.position.x < 0 || ball.position.x > canvas.width || ball.position.y < 0 || ball.position.y > canvas.height) {
        hasLost = true;
        isPlaying = false;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBackground();
    drawGameObjects();
    requestAnimationFrame(gameLoop);
}

function drawGameObjects() {
    drawCircle(ball);
    drawRectangle(target);

    engine.world.bodies
      .filter(body => body.label === 'attractor')
      .forEach(attractor => drawCircle(attractor, attractor.charge == ATTRACTOR_CHARGE ? 'red' : 'blue'));

    if (isMouseDown) {
        drawLaunchGuide(mouseCurrentPosition);
    }
    drawCounters();
}
function drawCircle(body, color = 'black') {
    const position = body.position;
    const radius = body.circleRadius;
    ctx.beginPath();
    ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawRectangle(body) {
    const position = body.position;
    const width = body.bounds.max.x - body.bounds.min.x;
    const height = body.bounds.max.y - body.bounds.min.y;
    ctx.fillRect(position.x - width / 2, position.y - height / 2, width, height);
}

function drawBackground() {
    ctx.fillStyle = 'green';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.beginPath();
    ctx.arc(centerX, centerY, 70, 0, 2 * Math.PI);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();


}

function drawCounters() {
    ctx.font = '20px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    ctx.fillText(`Deaths: ${deathCount}`, canvas.width - 10, 30);
    ctx.fillText(`Level: ${levelCount}`, canvas.width - 10, 60);
}

document.getElementById('reset-level').addEventListener('click', () => {
    resetGame();
  });
  
  document.getElementById('reset-game').addEventListener('click', () => {
    currentLevel = 0;
    deathCount = 0;
    levelCount = 1
    resetGame();
  });
  
  // Add touch event listeners for mobile devices
  document.getElementById('reset-level').addEventListener('touchstart', (event) => {
    event.preventDefault();
    resetLevel();
  });
  
  document.getElementById('reset-game').addEventListener('touchstart', (event) => {
    event.preventDefault();
    currentLevel = 0;
    deathCount = 0;
    levelCount = 1
    resetGame();
  });

  document.getElementById('instructions-button').addEventListener('click', () => {
    const instructionsElement = document.getElementById('instructions');
    if (instructionsElement.style.display === 'none') {
        instructionsElement.style.display = 'block';
        document.getElementById('instructions-button').textContent = 'Hide Instructions';
    } else {
        instructionsElement.style.display = 'none';
        document.getElementById('instructions-button').textContent = 'Show Instructions';
    }
});

setupLevel(currentLevel);
gameLoop();

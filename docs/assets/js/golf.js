const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const { Engine, Render, World, Bodies, Events } = Matter;

const engine = Engine.create();
engine.gravity = { x: 0, y: 0 }

const ball = Bodies.circle(50, canvas.height / 2, 10, { restitution: 0.5, label: 'ball' });
const target = Bodies.rectangle(canvas.width - 50, canvas.height / 2, 20, 80, { isStatic: true, label: 'target' });

World.add(engine.world, [ball, target]);

const K = 200, ATTRACTOR_CHARGE = 1, REPELLER_CHARGE = -0.3;
const ATTRACTOR_AMPLITUDE = 100, REPELLER_AMPLITUDE = 0;
const ATTRACTOR_SPEED = -12.5, REPELLER_SPEED = 12.5;
const MAX_INITIAL_SPEED = 2;
let currentLevel = 0;
let deathCount = 0;
let levelCount = 1;

function setupLevel(level) {
    engine.world.bodies
        .filter(body => body.label === 'attractor')
        .forEach(attractor => World.remove(engine.world, attractor));

    for (let i = 0; i < 2*level + 1; i++) {
        const attractor = Bodies.circle(200 + 100 * i, 200, 10, { isStatic: true, label: 'attractor' });
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

    engine.world.bodies
        .filter(body => body.label === 'attractor')
        .forEach((attractor, i) => {
            const baseSpeed =  (attractor.charge == ATTRACTOR_CHARGE ? ATTRACTOR_SPEED : REPELLER_SPEED); 
            const baseAmplitude = (attractor.charge == ATTRACTOR_CHARGE ? ATTRACTOR_AMPLITUDE : REPELLER_AMPLITUDE);
            const speed = baseSpeed * (i % 2 === 0 ? 1 : -1);
            attractor.position.y = 200 + Math.sin(t * speed) * baseAmplitude;
        });
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
canvas.addEventListener('mousedown', (event) => {
    isStarted = true;
    isMouseDown = true;
    mouseDownPosition = { x: event.clientX, y: event.clientY };
});

canvas.addEventListener('mouseup', (event) => {
    if (isMouseDown) {
        const mouseUpPosition = { x: event.clientX, y: event.clientY };
        launchVector = {
            x: -(mouseDownPosition.x - mouseUpPosition.x) * 0.1,
            y: -(mouseDownPosition.y - mouseUpPosition.y) * 0.1
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
});

canvas.addEventListener('mousemove', (event) => {
    mouseCurrentPosition = { x: event.clientX, y: event.clientY };
});

function drawLaunchGuide(currentPosition) {
    ctx.beginPath();
    ctx.moveTo(mouseDownPosition.x - 270 , mouseDownPosition.y - 90);
    ctx.lineTo(currentPosition.x - 270, currentPosition.y - 90);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function resetGame() {
    if (hasLost) {
        deathCount++;
    }
    Matter.Body.setPosition(ball, { x: 50, y: canvas.height / 2 });
    Matter.Body.setVelocity(ball, { x: 0, y: 0 });

    isStarted = false;
    isPlaying = false;
    hasWon = false;
    hasLost = false;

    setupLevel(currentLevel);
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
    drawGameObjects();
    requestAnimationFrame(gameLoop);
}

function drawGameObjects() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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

function drawCounters() {
    ctx.font = '20px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    ctx.fillText(`Deaths: ${deathCount}`, canvas.width - 10, 30);
    ctx.fillText(`Level: ${levelCount}`, canvas.width - 10, 60);
}

setupLevel(currentLevel);
gameLoop();

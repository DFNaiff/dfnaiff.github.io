const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

let player;

//player.js
class Player {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.width = 10;
    this.height = 10;
    this.speed = 2;
    this.lives = 3;
    this.bullets = [];
  }

  move(direction) {
    if (direction === 'left') {
      this.x -= this.speed;
    } else if (direction === 'right') {
      this.x += this.speed;
    }

    // Keep the player within the canvas boundaries
    this.x = Math.max(0, Math.min(this.x, canvas.width - this.width));
  }

  update() {
    // You can add more logic here, like shooting and taking damage
  }

  draw() {
    ctx.fillStyle = 'white';
    ctx.fillRect(this.x, this.y, this.width, this.height);
  }
}

//main.js
function init() {
  // Create a player instance at the bottom center of the canvas
  player = new Player(canvas.width / 2 - 5, canvas.height - 20, ctx);

  // Initialize the game loop
  gameLoop();
}

function restart() {
    // Reset game state (e.g., player lives, score, etc.)
    // ...
  
    // Call the init function to restart the game
    init();
  }

function update() {
  // Update the player
  player.update();
}

function draw() {
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the player
  player.draw();
}

function gameLoop() {
  update();
  draw();
  requestAnimationFrame(gameLoop);
}

// Add event listeners for keydown and keyup events to handle player movement
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') {
    player.move('left');
  } else if (e.key === 'ArrowRight') {
    player.move('right');
  }
});

// Call the init function to start the game
init();
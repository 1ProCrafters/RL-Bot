const { SERVER_CONFIG, TRAINING_CONFIG, ARENA_CONFIG } = require('./config');
const TrainingSystem = require('./TrainingSystem');

// ==================== MAIN ====================
async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const loadModel = args.includes('--load') || args.includes('-l') || args.includes('--continue') || args.includes('-c');

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
╔════════════════════════════════════════════════════════════╗
║     Minecraft RL PvP Bot Training System                  ║
╚════════════════════════════════════════════════════════════╝

Usage: node index.js [options]

Options:
  --load, -l         Load existing model and continue training
  --continue, -c     Same as --load
  --help, -h         Show this help message

Examples:
  node index.js                    # Start fresh training
  node index.js --continue         # Continue from saved model
  node index.js --load             # Load and continue training

Configuration:
  Edit TRAINING_CONFIG at the top of the file to adjust:
  
  Training:
    - numOperators: ${TRAINING_CONFIG.numOperators}
    - episodesPerSession: ${TRAINING_CONFIG.episodesPerSession}
    - saveInterval: ${TRAINING_CONFIG.saveInterval} (saves every N episodes)
    - modelPath: ${TRAINING_CONFIG.modelPath}
  
  Learning:
    - learningRate: ${TRAINING_CONFIG.learningRate}
    - explorationRate: ${TRAINING_CONFIG.explorationRate} → ${TRAINING_CONFIG.explorationMin}
    - explorationDecay: ${TRAINING_CONFIG.explorationDecay}
    - batchSize: ${TRAINING_CONFIG.batchSize}
  
  Rewards:
    - Kill: +${TRAINING_CONFIG.rewardKill}
    - Hit: +${TRAINING_CONFIG.rewardHit}
    - Damage Dealt: +${TRAINING_CONFIG.rewardDamageDealt} per HP
    - Death: ${TRAINING_CONFIG.rewardDeath}
    - Damage Taken: ${TRAINING_CONFIG.rewardDamageTaken} per HP

Server:
  - Host: ${SERVER_CONFIG.host}
  - Port: ${SERVER_CONFIG.port}
  - Version: ${SERVER_CONFIG.version}

Note: All Operator bots must have OP permissions on the server!
    `);
    process.exit(0);
  }

  const trainer = new TrainingSystem(loadModel, SERVER_CONFIG, TRAINING_CONFIG, ARENA_CONFIG);

  process.on('SIGINT', () => {
    console.log('\n\n⚠️  Interrupt received! Saving model...');
    trainer.saveAgent();
    console.log('✅ Model saved. Shutting down...');
    trainer.cleanup();
    process.exit(0);
  });

  process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught exception:', error);
    console.log('Saving model before exit...');
    trainer.saveAgent();
    trainer.cleanup();
    process.exit(1);
  });

  try {
    await trainer.train();
    trainer.cleanup();
  } catch (error) {
    console.error('❌ Error during training:', error);
    console.log('Saving model before exit...');
    trainer.saveAgent();
    trainer.cleanup();
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}
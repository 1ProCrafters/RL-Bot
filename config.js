// ==================== CONFIGURATION ====================
const SERVER_CONFIG = {
    host: 'localhost',
    port: 25565,
    version: '1.19.4'
};

const TRAINING_CONFIG = {
    numBots: 10,
    numOperators: 2, // Number of operator bots to use
    connectDelay: 100,
    episodesPerSession: 100,
    maxStepsPerEpisode: 600,
    learningRate: 0.01,
    discountFactor: 0.99,
    explorationRate: 1.0,
    explorationDecay: 0.9995,
    explorationMin: 0.05,

    // Rewards
    rewardKill: 200,
    rewardHit: 20,
    rewardDamageDealt: 3,
    rewardDeath: -200,
    rewardDamageTaken: -3,
    rewardProximity: 1, // Reward for being closer to opponent
    rewardFarAway: -1.5, // Penalty for being too far

    // Training
    saveInterval: 10, // Save model every N episodes
    modelPath: './model/pvp_model.json',
    logInterval: 1, // Log stats every N episodes
    targetNetworkUpdateInterval: 5,
    batchSize: 32,
    minMemorySize: 50,

    // Genetic Algorithm / Evolution
    enableEvolution: true, // Enable evolutionary training
    evolutionInterval: 2, // Every N episodes, evolve the population
    mutationRate: 0.1, // Probability of mutating each weight
    mutationStrength: 0.2, // How much to mutate weights by

    // Fitness metric formula (customize this!)
    // Available variables: kills, deaths, damageDealt, damageTaken, avgReward
    // Example: "kills - deaths * 0.5 + damageDealt * 0.1"
    fitnessFormula: "kills * 100 - deaths * 50 + damageDealt * 2 - damageTaken * 1",

    // Combat
    attackCooldown: 500, // ms between attacks
    optimalDistance: 3.5, // Optimal attack distance

    // Arena scaling
    botsPerArena: 2, // How many bots fight in each arena (must be 2 for now)
    arenaSpacing: 30, // Distance between arenas for multiple bot pairs
};

const ARENA_CONFIG = {
    center: { x: 100, y: -60, z: 100 },
    size: 15,
    respawnY: -57
};

module.exports = { SERVER_CONFIG, TRAINING_CONFIG, ARENA_CONFIG };
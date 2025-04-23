const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const logger = require('./logger');

/**
 * Set up the training environment by making the script executable
 * and creating the virtual environment if it doesn't exist
 */
function setupTrainingEnv() {
  try {
    const trainDir = path.resolve(__dirname, '../../../train');
    const setupScriptPath = path.join(trainDir, 'setup_env.sh');
    
    // Make the setup script executable
    fs.chmodSync(setupScriptPath, 0o755); 
    logger.info('Setup script permissions updated');
    
    // Check if we need to create the virtual environment
    if (!fs.existsSync(path.join(trainDir, 'venv'))) {
      logger.info('Creating virtual environment for training...');
      
      // Execute the setup script
      execSync(`${setupScriptPath}`, { 
        cwd: trainDir,
        stdio: 'inherit' 
      });
      
      logger.info('Training environment setup complete');
    } else {
      logger.info('Training environment already exists');
    }
    
    return true;
  } catch (error) {
    logger.error(`Failed to set up training environment: ${error.message}`);
    return false;
  }
}

module.exports = setupTrainingEnv;

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const logger = require('./logger');
const os = require('os');

/**
 * Set up the training environment by making the script executable
 * and creating the virtual environment if it doesn't exist
 */
function setupTrainingEnv() {
  try {
    const trainDir = path.resolve(__dirname, '../train');
    const setupScriptPath = path.join(trainDir, 'setup_env.sh');
    const venvPath = path.join(trainDir, 'venv');
    
    // Check if we need to create the virtual environment
    if (!fs.existsSync(venvPath)) {
      logger.info('Creating virtual environment for training...');
      
      // Platform-specific execution
      if (os.platform() === 'win32') {
        // On Windows, create the virtual environment directly
        logger.info('Running on Windows - executing Python commands directly');
        
        // Create virtual environment
        execSync('python -m venv venv', { 
          cwd: trainDir,
          stdio: 'inherit' 
        });
        
        // Install dependencies (using the Python executable to run pip)
        try {
          logger.info('Upgrading pip...');
          execSync('venv\\Scripts\\python -m pip install --upgrade pip', { 
            cwd: trainDir,
            stdio: 'inherit' 
          });
          
          logger.info('Installing dependencies...');
          // Install PyTorch separately to ensure it works
          execSync('venv\\Scripts\\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu', {
            cwd: trainDir,
            stdio: 'inherit'
          });
          
          // Install other requirements
          execSync('venv\\Scripts\\python -m pip install -r requirements.txt --no-deps', { 
            cwd: trainDir,
            stdio: 'inherit' 
          });
          
          // Verify installation
          logger.info('Verifying torch installation...');
          execSync('venv\\Scripts\\python -c "import torch; print(f\'PyTorch {torch.__version__} installed successfully\')"', {
            cwd: trainDir,
            stdio: 'inherit'
          });
        } catch (err) {
          logger.error(`Failed to install dependencies: ${err.message}`);
          throw new Error('Dependency installation failed');
        }
      } else {
        // On Unix-like systems, make the script executable and run it
        fs.chmodSync(setupScriptPath, 0o755); 
        logger.info('Setup script permissions updated');
        
        // Execute the setup script
        execSync(`${setupScriptPath}`, { 
          cwd: trainDir,
          stdio: 'inherit' 
        });
      }
      
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
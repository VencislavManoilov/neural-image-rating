const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const logger = require('./logger');
const os = require('os');
const setupTrainingEnv = require('./setupTrainingEnv');

/**
 * Execute the training script for a specific label
 * 
 * @param {string} labelName - The name of the label to train on
 * @param {string} datasetUrl - The URL of the dataset API
 * @returns {Promise<Object>} - Promise resolving to training result
 */
function trainModel(labelName, datasetUrl) {
  return new Promise((resolve, reject) => {
    // Path to the training script
    const trainScriptPath = path.resolve(__dirname, '../train/train.py');
    const trainDir = path.dirname(trainScriptPath);
    const venvPath = path.join(trainDir, 'venv');
    
    // Log the training start
    logger.info(`Starting training for label: ${labelName}`);
    logger.info(`Using dataset URL: ${datasetUrl}`);
    
    // Check if virtual environment exists
    if (!fs.existsSync(venvPath)) {
      logger.info('Virtual environment not found, attempting to set it up...');
      if (!setupTrainingEnv()) {
        logger.error('Failed to set up training environment');
        reject({
          success: false,
          label: labelName,
          message: 'Failed to set up Python virtual environment'
        });
        return;
      }
    }
    
    // Use python from virtual environment
    let pythonExecutable;
    if (process.platform === 'win32') {
      pythonExecutable = path.join(trainDir, 'venv', 'Scripts', 'python.exe');
    } else {
      pythonExecutable = path.join(trainDir, 'venv', 'bin', 'python');
    }
    
    // Check if the virtual environment python exists
    if (!fs.existsSync(pythonExecutable)) {
      logger.error(`Python executable not found at: ${pythonExecutable}`);
      
      // Fallback to system Python as a last resort
      const systemPython = process.platform === 'win32' ? 'python' : 'python3';
      logger.info(`Attempting to use system Python: ${systemPython}`);
      
      // Spawn python to execute the script with system Python
      const trainProcess = spawn(systemPython, [
        trainScriptPath,
        '--label', labelName
      ], {
        cwd: trainDir,
        env: { ...process.env }
      });
      
      let stdout = '';
      let stderr = '';
      
      // Collect stdout data
      trainProcess.stdout.on('data', (data) => {
        const dataStr = data.toString();
        stdout += dataStr;
        logger.info(`Training output: ${dataStr.trim()}`);
      });
      
      // Collect stderr data
      trainProcess.stderr.on('data', (data) => {
        const dataStr = data.toString();
        stderr += dataStr;
        logger.error(`Training error: ${dataStr.trim()}`);
      });
      
      // Handle process completion
      trainProcess.on('close', (code) => {
        if (code === 0) {
          logger.info(`Training completed successfully for label: ${labelName}`);
          resolve({
            success: true,
            label: labelName,
            message: 'Training completed successfully',
            output: stdout
          });
        } else {
          logger.error(`Training failed for label: ${labelName} with exit code ${code}`);
          reject({
            success: false,
            label: labelName,
            message: 'Training failed',
            error: stderr,
            output: stdout,
            code
          });
        }
      });
      
      // Handle process errors
      trainProcess.on('error', (err) => {
        logger.error(`Failed to start training process: ${err.message}`);
        reject({
          success: false,
          label: labelName,
          message: 'Failed to start training process',
          error: err.message
        });
      });
      
      return;
    }
    
    logger.info(`Using Python interpreter: ${pythonExecutable}`);
    
    // Spawn python to execute the script
    const trainProcess = spawn(pythonExecutable, [
      trainScriptPath,
      '--label', labelName
    ], {
      cwd: trainDir,
      env: { ...process.env }
    });
    
    let stdout = '';
    let stderr = '';
    
    // Collect stdout data
    trainProcess.stdout.on('data', (data) => {
      const dataStr = data.toString();
      stdout += dataStr;
      logger.info(`Training output: ${dataStr.trim()}`);
    });
    
    // Collect stderr data
    trainProcess.stderr.on('data', (data) => {
      const dataStr = data.toString();
      stderr += dataStr;
      logger.error(`Training error: ${dataStr.trim()}`);
    });
    
    // Handle process completion
    trainProcess.on('close', (code) => {
      if (code === 0) {
        logger.info(`Training completed successfully for label: ${labelName}`);
        resolve({
          success: true,
          label: labelName,
          message: 'Training completed successfully',
          output: stdout
        });
      } else {
        logger.error(`Training failed for label: ${labelName} with exit code ${code}`);
        reject({
          success: false,
          label: labelName,
          message: 'Training failed',
          error: stderr,
          output: stdout,
          code
        });
      }
    });
    
    // Handle process errors
    trainProcess.on('error', (err) => {
      logger.error(`Failed to start training process: ${err.message}`);
      reject({
        success: false,
        label: labelName,
        message: 'Failed to start training process',
        error: err.message
      });
    });
  });
}

module.exports = trainModel;

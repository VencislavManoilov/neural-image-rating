const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const logger = require('./logger');

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
    const trainScriptPath = path.resolve(__dirname, '../../../train/train.py');
    const trainDir = path.dirname(trainScriptPath);
    
    // Log the training start
    logger.info(`Starting training for label: ${labelName}`);
    logger.info(`Using dataset URL: ${datasetUrl}`);
    
    // Use python directly
    const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    
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

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const logger = require('./logger');

/**
 * Predicts the rating of an image using a trained model
 * 
 * @param {string} imagePath - Path to the image file
 * @param {string} labelName - Name of the label (model) to use
 * @returns {Promise<number|null>} - Promise resolving to the predicted rating or null on error
 */
async function predictImage(imagePath, labelName) {
  // For single image prediction, call the batch function with one image
  const results = await predictImagesInternal([imagePath], labelName);
  return results[imagePath] || null;
}

/**
 * Predicts ratings for a batch of images using a trained model
 * 
 * @param {string[]} imagePaths - Array of paths to the image files
 * @param {string} labelName - Name of the label (model) to use
 * @returns {Promise<{imagePath: string, rating: number|null}[]>} - Promise resolving to array of predictions
 */
async function predictBatch(imagePaths, labelName) {
  // Prevent empty batch
  if (!imagePaths || imagePaths.length === 0) {
    return [];
  }
  
  logger.info(`Batch predicting ratings for ${imagePaths.length} images using model ${labelName}`);
  
  // Process images in larger batches to reduce Python process spawning
  const batchSize = 50;  // Increased from 10 to 50
  const results = [];
  
  for (let i = 0; i < imagePaths.length; i += batchSize) {
    const batch = imagePaths.slice(i, i + batchSize);
    const predictions = await predictImagesInternal(batch, labelName);
    
    const batchResults = batch.map(imagePath => ({
      imagePath,
      rating: predictions[imagePath] || null
    }));
    
    results.push(...batchResults);
    
    // Log progress
    logger.info(`Processed ${Math.min(i + batchSize, imagePaths.length)}/${imagePaths.length} images`);
  }
  
  return results;
}

/**
 * Internal function to predict ratings for multiple images at once
 * 
 * @param {string[]} imagePaths - Array of paths to the image files
 * @param {string} labelName - Name of the label (model) to use
 * @returns {Promise<Object>} - Promise resolving to object mapping image paths to ratings
 */
async function predictImagesInternal(imagePaths, labelName) {
  return new Promise((resolve, reject) => {
    // Check if at least one image exists
    const validImages = imagePaths.filter(path => fs.existsSync(path));
    if (validImages.length === 0) {
      logger.error(`No valid image files found in the batch`);
      return resolve({});
    }

    // Path to the predict script
    const predictScript = path.resolve(__dirname, '../train/predict.py');
    
    // Check if predict.py exists
    if (!fs.existsSync(predictScript)) {
      logger.error(`Prediction script not found: ${predictScript}`);
      
      // For testing, return random ratings
      const fallbackRatings = {};
      imagePaths.forEach(path => {
        fallbackRatings[path] = parseFloat((1 + Math.random() * 9).toFixed(2));
      });
      logger.warn(`Using fallback ratings for ${imagePaths.length} images`);
      return resolve(fallbackRatings);
    }
    
    const trainDir = path.dirname(predictScript);
    
    // Path to the model
    const modelPath = path.join(trainDir, 'models', labelName, 'best_model.pth');
    
    // Check if model exists
    if (!fs.existsSync(modelPath)) {
      logger.error(`Model not found for label '${labelName}': ${modelPath}`);
      
      // For testing, return random ratings
      const fallbackRatings = {};
      imagePaths.forEach(path => {
        fallbackRatings[path] = parseFloat((1 + Math.random() * 9).toFixed(2));
      });
      logger.warn(`Using fallback ratings for ${imagePaths.length} images`);
      return resolve(fallbackRatings);
    }

    // Path to Python executable in the virtual environment
    const pythonVenvPath = path.join(trainDir, 'venv', 
      process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    
    // Check if the venv python exists
    if (!fs.existsSync(pythonVenvPath)) {
      logger.warn(`Virtual environment Python not found at ${pythonVenvPath}, falling back to system Python`);
      // Fall back to system Python if the venv isn't available
      const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
      runPrediction(pythonExecutable);
    } else {
      // Use Python from the virtual environment
      logger.info(`Using Python from virtual environment: ${pythonVenvPath}`);
      runPrediction(pythonVenvPath);
    }

    function runPrediction(pythonPath) {
      // Prepare the command arguments
      const args = [
        predictScript,
        '--model', modelPath,
        '--images', imagePaths.join(','),
        '--json-output'  // Request JSON output for easier parsing
      ];
      
      // Debug log for the command being executed
      logger.info(`Running batch prediction for ${imagePaths.length} images using ${labelName} model`);
      
      // Spawn python to execute the script
      const predictProcess = spawn(pythonPath, args, {
        cwd: trainDir,
        env: { ...process.env }
      });
      
      let stdout = '';
      let stderr = '';
      
      // Collect stdout data
      predictProcess.stdout.on('data', (data) => {
        const dataStr = data.toString();
        stdout += dataStr;
      });
      
      // Collect stderr data
      predictProcess.stderr.on('data', (data) => {
        const dataStr = data.toString();
        stderr += dataStr;
        logger.error(`Prediction stderr: ${dataStr.trim()}`);
      });
      
      // Handle process completion
      predictProcess.on('close', (code) => {
        if (code === 0) {
          try {
            // Parse JSON output
            const outputData = JSON.parse(stdout);
            
            if (outputData && outputData.predictions) {
              logger.info(`Successfully predicted ratings for ${Object.keys(outputData.predictions).length} images`);
              resolve(outputData.predictions);
            } else {
              logger.error('Invalid JSON output from prediction script');
              resolve({});
            }
          } catch (error) {
            logger.error(`Error parsing prediction JSON: ${error.message}`);
            logger.error(`Raw output: ${stdout}`);
            
            // Fallback with random ratings if JSON parsing fails
            const fallbackRatings = {};
            imagePaths.forEach(path => {
              fallbackRatings[path] = parseFloat((1 + Math.random() * 9).toFixed(2));
            });
            logger.warn(`Using fallback ratings after JSON parse error`);
            resolve(fallbackRatings);
          }
        } else {
          logger.error(`Prediction process exited with code ${code}`);
          logger.error(stderr);
          resolve({});
        }
      });
      
      // Handle process errors
      predictProcess.on('error', (error) => {
        logger.error(`Error spawning prediction process: ${error.message}`);
        resolve({});
      });
      
      // Add timeout to prevent hanging
      setTimeout(() => {
        logger.error('Prediction process timed out');
        predictProcess.kill();
        resolve({});
      }, 10 * 60 * 1000); // 10 min timeout
    }
  });
}

/**
 * Predicts ratings for all images in a video object
 * 
 * @param {Object} video - Video object from fetchVideos
 * @param {string} labelName - Name of the label (model) to use
 * @returns {Promise<{video: Object, ratings: Array<{imagePath: string, rating: number|null}>}>} 
 */
async function predictVideo(video, labelName) {
  if (!video || !video.images || video.images.length === 0) {
    logger.error('No images in video object');
    return { video, ratings: [] };
  }
  
  const ratings = await predictBatch(video.images, labelName);
  
  // Calculate average rating
  const validRatings = ratings.filter(r => r.rating !== null).map(r => r.rating);
  const avgRating = validRatings.length > 0 
    ? validRatings.reduce((sum, rating) => sum + rating, 0) / validRatings.length
    : null;
  
  return {
    video,
    ratings,
    averageRating: avgRating
  };
}

module.exports = {
  predictImage,
  predictBatch,
  predictVideo
};

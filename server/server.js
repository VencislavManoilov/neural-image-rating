const express = require('express');
const app = express();
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cheerio = require('cheerio');
const knex = require('./knex');
const ensureSchema = require('./schema');
const logger = require('./utils/logger');
const setupTrainingEnv = require('./utils/setupTrainingEnv');
const cors = require('cors');
const Authorization = require('./middleware/Authorization');
const fetchVideos = require('./utils/fetchVideos');
const { predictImage, predictBatch, predictVideo } = require('./utils/predictImage');

const dotenv = require('dotenv');
dotenv.config();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const PORT = process.env.PORT || 8080;

const SITE_URL = process.env.SITE_URL;

const allowedOrigins = process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',') : ['http://localhost:3000'];
app.use(cors({
    origin: allowedOrigins,
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
}));

// Create thumbnails directory if it doesn't exist
const thumbnailsDir = path.join(__dirname, 'thumbnails');
if (!fs.existsSync(thumbnailsDir)) {
    fs.mkdirSync(thumbnailsDir, { recursive: true });
}

async function getThumbnails(page) {
    try {
        // Fetch the page content
        const response = await axios.get(SITE_URL+"/video?page="+page);
        if (response.status !== 200) {
            throw new Error(`Failed to fetch page: ${response.status}`);
        }
        
        const html = response.data;
        
        // Parse HTML with cheerio
        const $ = cheerio.load(html);
        const thumbnailUrls = [];
        
        // Find all elements with data-mediumthumb attribute
        $('[data-mediumthumb]').each((i, element) => {
            const thumbnailUrl = $(element).attr('data-mediumthumb');
            if (thumbnailUrl && thumbnailUrl.startsWith('http')) {
                thumbnailUrls.push(thumbnailUrl);
            }
        });
        
        logger.info(`Found ${thumbnailUrls.length} thumbnail URLs`);
        
        // Download each thumbnail
        const downloadPromises = thumbnailUrls.map(async (url, index) => {
            try {
                const imageResponse = await axios({
                    url,
                    method: 'GET',
                    responseType: 'stream'
                });
                
                // Extract filename from URL or create a numbered filename
                const filename = url.split('/').pop().split('?')[0] || `thumbnail-${index}.jpg`;
                const filepath = path.join(thumbnailsDir, filename);
                
                // Create a write stream and pipe the image data to it
                const writer = fs.createWriteStream(filepath);
                imageResponse.data.pipe(writer);
                
                return new Promise((resolve, reject) => {
                    writer.on('finish', () => resolve(filepath));
                    writer.on('error', reject);
                });
            } catch (error) {
                logger.error(`Error downloading thumbnail ${url}:` + error.message);
                return null;
            }
        });
        
        const downloadedFiles = await Promise.all(downloadPromises);
        const successfulDownloads = downloadedFiles.filter(file => file !== null);
        
        logger.info(`Successfully downloaded ${successfulDownloads.length} thumbnails`);
        return successfulDownloads;
    } catch (error) {
        logger.error("Error fetching thumbnails:" + error);
        return [];
    }
}

app.get("/", (req, res) => {
    res.status(200).json({
        message: "Welcome to the HotBot API",
        version: "1.0.0",
    });
});

const authRoute = require("./routes/auth");
app.use("/auth", authRoute);

const labelsRoute = require("./routes/labels");
app.use("/labels", Authorization, labelsRoute);

// Add a route to trigger thumbnail downloading
app.get("/download-thumbnails", async (req, res) => {
    const page = req.query.page || 1;

    try {
        const thumbnails = await getThumbnails(page);
        res.status(200).json({
            message: `Successfully downloaded ${thumbnails.length} thumbnails`,
            // thumbnails
        });
    } catch (error) {
        res.status(500).json({
            message: "Error downloading thumbnails",
            error: error.message
        });
    }
});

app.get("/fetch-videos", async (req, res) => {
    try {
        // Get optional label parameter for rating predictions
        const label = req.query.label;
        
        logger.info(`Fetching videos${label ? ` with ratings using label: ${label}` : ''}`);
        
        const html = await axios.get(SITE_URL, {
            headers: {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': 'Linux',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
            }
        });
        const videos = new fetchVideos(html.data, label);

        // Await the fetchVideos method to complete
        const fetchedVideos = await videos.fetchVideos();

        // Log the number of videos with ratings for debugging
        if (label) {
            const videosWithRatings = fetchedVideos.filter(v => 
                v.ratings && v.ratings.length > 0 && v.ratings.some(r => r !== null)
            );
            
            logger.info(`Found ${videosWithRatings.length} videos with non-empty ratings out of ${fetchedVideos.length} total videos`);
        }

        const responseData = {
            message: `Successfully fetched ${fetchedVideos.length} videos with ${fetchedVideos.reduce((sum, video) => sum + video.images.length, 0)} images`,
            videos: fetchedVideos
        };
        
        // Include rating information if ratings were made
        if (label) {
            responseData.label = label;
            const videosWithRatings = fetchedVideos.filter(v => v.averageRating !== null);
            responseData.averageRating = videosWithRatings.length > 0 
                ? videosWithRatings.reduce((sum, v) => sum + v.averageRating, 0) / videosWithRatings.length 
                : 0;
        }

        res.status(200).json(responseData);
    } catch (error) {
        logger.error(`Error fetching videos: ${error.message}`);
        res.status(500).json({
            message: "Error fetching videos",
            error: error.message
        });
    }
});

app.post("/predict-image", Authorization, async (req, res) => {
    try {
        const { imagePath, label } = req.body;
        
        if (!imagePath || !label) {
            return res.status(400).json({
                message: "Missing required parameters: imagePath and label"
            });
        }
        
        const rating = await predictImage(imagePath, label);
        
        if (rating === null) {
            return res.status(500).json({
                message: "Failed to predict rating"
            });
        }
        
        res.status(200).json({
            message: "Successfully predicted rating",
            imagePath,
            label,
            rating
        });
    } catch (error) {
        logger.error("Error predicting image rating:" + error);
        res.status(500).json({
            message: "Error predicting image rating",
            error: error.message
        });
    }
});

app.post("/predict-videos", Authorization, async (req, res) => {
    try {
        const { videoUrl: videosUrl, label } = req.body;
        
        if (!videosUrl || !label) {
            return res.status(400).json({
                message: "Missing required parameters: videoUrl and label"
            });
        }
        
        // Fetch video content
        const html = await axios.get(videosUrl);
        const videoFetcher = new fetchVideos(html.data);
        
        // Extract video frames
        const fetchedVideos = await videoFetcher.fetchVideos();
        
        if (!fetchedVideos || fetchedVideos.length === 0) {
            return res.status(404).json({
                message: "No videos found in the provided URL"
            });
        }
        
        // Predict ratings for all videos
        const predictions = await Promise.all(
            fetchedVideos.map(video => predictVideo(video, label))
        );
        
        // Calculate overall rating as average of all video average ratings
        const validPredictions = predictions.filter(p => p.averageRating !== null);
        const overallRating = validPredictions.length > 0
            ? validPredictions.reduce((sum, p) => sum + p.averageRating, 0) / validPredictions.length
            : null;
        
        res.status(200).json({
            message: "Successfully predicted ratings for videos",
            overallRating,
            predictions: predictions.map(p => ({
                averageRating: p.averageRating,
                imageCount: p.ratings.length,
                validRatings: p.ratings.filter(r => r.rating !== null).length
            }))
        });
    } catch (error) {
        logger.error("Error predicting video ratings:" + error);
        res.status(500).json({
            message: "Error predicting video ratings",
            error: error.message
        });
    }
});

app.post("/predict-batch", Authorization, async (req, res) => {
    try {
        const { imagePaths, label } = req.body;
        
        if (!imagePaths || !Array.isArray(imagePaths) || !label) {
            return res.status(400).json({
                message: "Missing or invalid parameters: imagePaths should be an array and label is required"
            });
        }
        
        const ratings = await predictBatch(imagePaths, label);
        
        // Calculate average of valid ratings
        const validRatings = ratings.filter(r => r.rating !== null);
        const averageRating = validRatings.length > 0
            ? validRatings.reduce((sum, r) => sum + r.rating, 0) / validRatings.length
            : null;
        
        res.status(200).json({
            message: "Successfully predicted ratings for batch",
            averageRating,
            ratings
        });
    } catch (error) {
        logger.error("Error predicting batch ratings:" + error);
        res.status(500).json({
            message: "Error predicting batch ratings",
            error: error.message
        });
    }
});

// Add a health check endpoint
app.get("/health", (req, res) => {
    res.status(200).json({ status: "ok" });
});

(async () => {
    try {
        ensureSchema().then(async () => {
            // Set up training environment
            setupTrainingEnv();
            
            app.listen(PORT, () => {
                logger.info(`Server is running on port ${PORT}`);
            });
        })
    } catch (error) {
        logger.error("Error ensuring database schema:" + error);
    }
})()
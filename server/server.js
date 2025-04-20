const express = require('express');
const app = express();
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cheerio = require('cheerio');
const knex = require('./knex');
const ensureSchema = require('./schema');
const logger = require('./utils/logger');

const dotenv = require('dotenv');
dotenv.config();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const PORT = process.env.PORT || 8080;

const SITE_URL = process.env.SITE_URL;

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
app.use("/auth", (req, res, next) => {
    req.knex = knex;
    next();
}, authRoute);

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

(async () => {
    try {
        ensureSchema().then(async () => {
            app.listen(PORT, () => {
                logger.info(`Server is running on port ${PORT}`);
            });
        })
    } catch (error) {
        logger.error("Error ensuring database schema:" + error);
    }
})()
const express = require('express');
const router = express.Router();
const knex = require('../knex');
const logger = require('../utils/logger');
const { default: axios } = require('axios');

const DATASET_URL = process.env.DATASET_URL || 'http://localhost:5000';

router.get('/me', async (req, res) => {
    try {
        const userId = req.user.id;
    
        const labels = await knex('labels')
        .where({ userId })
        .first();
    
        if(!labels) {
            return res.status(404).json({ message: 'Labels not found' });
        }
    
        res.json({
            message: 'Labels retrieved successfully',
            labels
        });
    } catch (error) {
        logger.error('Error retrieving labels:' + error);
        res.status(500).json({ message: 'Server error while retrieving labels' });
    }
});

router.get('/get/:name', async (req, res) => {
    try {
        const labelsName = req.params.name;

        const response = await axios.get(`${DATASET_URL}/labels/${labelsName}`);
        if (response.status !== 200) {
            return res.status(404).json({ message: 'Labels not found' });
        }

        const { labels } = response.data;

        res.json({
            message: 'Labels retrieved successfully',
            labels
        });
    } catch (error) {
        logger.error('Error retrieving labels:' + error);
        res.status(500).json({ message: 'Server error while retrieving labels' });
    }
});

router.post('/add', async (req, res) => {
    try {
        const response = await axios.post(`${DATASET_URL}/add`, {
            name: req.user.email
        });

        if (response.status !== 200) {
            return res.status(400).json({ message: 'Failed to add labels' });
        }

        await knex('labels').insert({
            userId: req.user.id,
            label: response.data.name
        });

        res.status(200).json({
            message: 'Labels added successfully',
            name: response.data.name
        });
    } catch (error) {
        logger.error('Error adding labels:' + error);
        res.status(500).json({ message: 'Server error while adding labels' });
    }
});

module.exports = router;
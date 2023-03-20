CREATE DATABASE ml;

USE ml;

CREATE TABLE
    `CHAT` (
        `id` INT AUTO_INCREMENT PRIMARY KEY,
        `uid` INT,
        `message` VARCHAR(255)
    )
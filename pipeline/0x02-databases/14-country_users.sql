-- creates a table users following these requirements:
-- id, integer, never null, auto increment and primary key
-- email, string (255 characters), never null and unique
-- name, string (255 characters)
-- country, enumeration of countries: US, CO and TN, never null
-- (= default will be the first element of the enumeration, here US)
-- If the table already exists, your script should not fail
CREATE TABLE IF NOT EXISTS users(
id INT NOT NULL AUTO_INCREMENT,
email VARCHAR(255) NOT NULL,
name VARCHAR(255),
country ENUM("US", "CO", "TN") DEFAULT "US",
UNIQUE(email),
PRIMARY KEY(id));

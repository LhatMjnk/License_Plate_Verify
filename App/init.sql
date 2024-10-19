CREATE TABLE IF NOT EXISTS parking_users (
    id_user SERIAL PRIMARY KEY,
    plate VARCHAR(255) NOT NULL,
    user_name VARCHAR(255) NOT NULL,
    msv INTEGER NOT NULL
);
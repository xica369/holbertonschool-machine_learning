-- creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0
-- The function SafeDiv takes 2 arguments:
-- a, INT
-- b, INT
-- Returns a / b or 0 if b == 0
DELIMITER //

CREATE FUNCTION SafeDiv(a INT,
       			b INT)

RETURNS FLOAT
BEGIN
  RETURN (IF (b = 0, 0, (a / b)));
END //
DELIMITER;

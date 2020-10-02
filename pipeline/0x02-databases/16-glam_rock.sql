-- lists all bands with Glam as their main style, ranked by their longevity
-- Column names must be: band_name and lifespan (in years)
-- You should use attributes formed and split for computing the lifespan
SELECT band_name, IF(split IS NULL, (2020-formed), (split - formed)) AS lifespan
FROM metal_bands
WHERE style REGEXP "Glam rock"
ORDER BY lifespan DESC;

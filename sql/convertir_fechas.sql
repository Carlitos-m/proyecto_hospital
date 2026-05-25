-- 1. Agregamos las columnas con el formato correcto de MySQL
ALTER TABLE hostransacciones ADD COLUMN fecha_real DATE;
ALTER TABLE hostransacciones ADD COLUMN fecha_hora_real DATETIME;

-- 2. Convertimos y llenamos los datos
-- Actualizamos toda la tabla (sin restricciones de cantidad)
UPDATE hostransacciones
SET fecha_real = STR_TO_DATE(CAST(fecha AS CHAR), '%Y%m%d')
WHERE fecha IS NOT NULL AND fecha <> '0';

UPDATE hostransacciones
SET fecha_hora_real = STR_TO_DATE(
        CONCAT(CAST(fecha AS CHAR), LPAD(CAST(hora AS CHAR), 6, '0')),
        '%Y%m%d%H%i%s'
    )
WHERE fecha IS NOT NULL AND hora IS NOT NULL AND fecha <> '0';

-- SELECT fecha, hora, fecha_real, fecha_hora_real
-- FROM hostransacciones
-- LIMIT 10;

-- para las demas tablas
ALTER TABLE hosffa ADD COLUMN f_fac_real DATE;
ALTER TABLE hosffa ADD COLUMN f_fac_hora_real DATETIME;

-- Ejemplo para hosffa (reemplaza 'f_fac' y 'h_fac' por los nombres de tus columnas reales)
UPDATE hosffa
SET f_fac_real = STR_TO_DATE(CAST(f_fac AS CHAR), '%Y%m%d')
WHERE f_fac IS NOT NULL AND f_fac <> '0';

UPDATE hosffa
SET f_fac_hora_real = STR_TO_DATE(
        CONCAT(CAST(f_fac AS CHAR), LPAD(CAST(h_fac AS CHAR), 6, '0')),
        '%Y%m%d%H%i%s'
    )
WHERE f_fac IS NOT NULL AND h_fac IS NOT NULL AND f_fac <> '0';



-- Preparación para hospac
ALTER TABLE hospac ADD COLUMN p_res_fec_real DATE;
ALTER TABLE hospac ADD COLUMN p_res_hra_real DATETIME;

-- Conversión para la tabla hospac
UPDATE hospac SET p_res_fec_real = STR_TO_DATE(CAST(p_res_fec AS CHAR), '%Y%m%d') WHERE p_res_fec IS NOT NULL AND p_res_fec <> '0';
UPDATE hospac SET p_res_hra_real = STR_TO_DATE(CONCAT(CAST(p_res_fec AS CHAR), LPAD(CAST(p_res_hra AS CHAR), 6, '0')), '%Y%m%d%H%i%s') WHERE p_res_fec IS NOT NULL AND p_res_hra IS NOT NULL AND p_res_fec <> '0';



-- Preparación
ALTER TABLE hosreq ADD COLUMN fec_sol_real DATE;
ALTER TABLE hosreq ADD COLUMN fec_sol_hora_real DATETIME;

-- Conversión
UPDATE hosreq SET fec_sol_real = STR_TO_DATE(CAST(fec_sol AS CHAR), '%Y%m%d') WHERE fec_sol IS NOT NULL AND fec_sol <> '0';
UPDATE hosreq SET fec_sol_hora_real = STR_TO_DATE(CONCAT(CAST(fec_sol AS CHAR), LPAD(CAST(hra_sol AS CHAR), 6, '0')), '%Y%m%d%H%i%s') WHERE fec_sol IS NOT NULL AND hra_sol IS NOT NULL AND fec_sol <> '0';



-- Preparación
ALTER TABLE hosfol ADD COLUMN f_fec_vig_real DATE;
ALTER TABLE hosfol ADD COLUMN f_fec_vig_hora_real DATETIME;

-- Conversión
UPDATE hosfol
SET f_fec_vig_hora_real = STR_TO_DATE(
        CONCAT(
                CAST(f_fec_vig AS CHAR),
                LPAD(
                        CASE
                            WHEN f_hra_vig REGEXP '^[0-9]+$' AND CAST(f_hra_vig AS UNSIGNED) <= 235959 THEN f_hra_vig
                            ELSE '0'
                            END,
                        6, '0'
                    )
            ),
        '%Y%m%d%H%i%s'
    )
WHERE f_fec_vig REGEXP '^[0-9]{8}$' -- Solo procesa si la fecha tiene exactamente 8 números
  AND CAST(f_fec_vig AS UNSIGNED) >= 19000101; -- Y que sea una fecha mayor a 1900


SELECT f_fec_vig, f_hra_vig
FROM hosfol
WHERE f_fec_vig <> '0'
  AND NOT (f_hra_vig REGEXP '^[0-9]+$')
  AND f_hra_vig IS NOT NULL;







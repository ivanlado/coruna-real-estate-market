update viviendas
set
    n_habitaciones = 1
where
    id in (
        SELECT
            id
        FROM
            viviendas
        WHERE
            LOWER(titulo) LIKE 'estudio%'
            AND n_habitaciones IS NULL
    )
CREATE OR REPLACE FUNCTION insert_vivienda(
    p_id integer,
    p_titulo text,
    p_n_habitaciones integer,
    p_tamano integer,
    p_descripcion text,
    p_extra_info text,
    p_img_file text,
    p_n_img integer,
    p_precio money,
    p_precio_modalidad varchar(5))
RETURNS boolean AS
$$
DECLARE
    success boolean := false;
BEGIN
    BEGIN
        -- Try to insert into viviendas table
        INSERT INTO public.viviendas(id, titulo, descripcion, extra_info, n_habitaciones, tamano, precio, precio_modalidad)
        VALUES (p_id, p_titulo, p_descripcion, p_extra_info, p_n_habitaciones, p_tamano, p_precio, p_precio_modalidad);
        
        success := true; -- Set success if viviendas insert succeeds
        -- Try to insert into imagenes table
        INSERT INTO public.imagenes(id, img_file, n_img)
        VALUES (p_id, p_img_file, p_n_img);
        
        -- Try to insert into registro table
        INSERT INTO public.registro(id, primera_fecha, ultima_fecha, fin_fecha)
        VALUES (p_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL);
        
    EXCEPTION
        WHEN unique_violation THEN
		BEGIN
            -- Handle the unique violation for viviendas here
            UPDATE public.registro SET ultima_fecha = CURRENT_TIMESTAMP WHERE id = p_id;
            success := false; -- Set success to false for viviendas insert
    	END;
	END;

    RETURN success; -- Return the overall success status
END;

$$
LANGUAGE plpgsql;
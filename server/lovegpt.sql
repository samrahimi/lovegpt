CREATE TABLE IF NOT EXISTS public.users (
    id serial PRIMARY KEY,
    username text,
	password text,
	email text,
	sex boolean,
	membership_type text default 'free',
	email_verified boolean default false,
	rate_limited boolean default false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);



ALTER TABLE IF EXISTS public.users
    OWNER to postgres;
--
-- PostgreSQL database dump
--

\restrict 6jyLz9nP6dFLLZZN83StpVCAkAzRq94OhoCJ2ws0iUAhevEzY6VlA5Tu4t88ZIp

-- Dumped from database version 14.19 (Ubuntu 14.19-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.19 (Ubuntu 14.19-0ubuntu0.22.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: banks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.banks (
    bank_id integer NOT NULL,
    bank_name character varying(100) NOT NULL,
    app_name character varying(200) NOT NULL,
    app_id character varying(200) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.banks OWNER TO postgres;

--
-- Name: TABLE banks; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.banks IS 'Stores information about Ethiopian banks and their mobile banking apps';


--
-- Name: banks_bank_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.banks_bank_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.banks_bank_id_seq OWNER TO postgres;

--
-- Name: banks_bank_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.banks_bank_id_seq OWNED BY public.banks.bank_id;


--
-- Name: reviews; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reviews (
    review_id integer NOT NULL,
    bank_id integer NOT NULL,
    review_text text NOT NULL,
    rating integer NOT NULL,
    review_date date NOT NULL,
    sentiment_label character varying(20),
    sentiment_score numeric(5,4),
    pos_score numeric(5,4),
    neu_score numeric(5,4),
    neg_score numeric(5,4),
    rating_adjusted boolean DEFAULT false,
    identified_themes text,
    preprocessed_text text,
    source character varying(50) DEFAULT 'Google Play'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT reviews_rating_check CHECK (((rating >= 1) AND (rating <= 5)))
);


ALTER TABLE public.reviews OWNER TO postgres;

--
-- Name: TABLE reviews; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.reviews IS 'Stores scraped and analyzed Google Play Store reviews with sentiment and thematic data';


--
-- Name: COLUMN reviews.bank_id; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.bank_id IS 'Foreign key reference to banks table';


--
-- Name: COLUMN reviews.review_text; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.review_text IS 'Original review text from Google Play Store';


--
-- Name: COLUMN reviews.rating; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.rating IS 'User rating (1-5 stars)';


--
-- Name: COLUMN reviews.review_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.review_date IS 'Date when the review was posted';


--
-- Name: COLUMN reviews.sentiment_label; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.sentiment_label IS 'Sentiment classification: Positive, Negative, or Neutral';


--
-- Name: COLUMN reviews.sentiment_score; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.sentiment_score IS 'VADER compound sentiment score (-1 to 1)';


--
-- Name: COLUMN reviews.pos_score; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.pos_score IS 'VADER positive sentiment score';


--
-- Name: COLUMN reviews.neu_score; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.neu_score IS 'VADER neutral sentiment score';


--
-- Name: COLUMN reviews.neg_score; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.neg_score IS 'VADER negative sentiment score';


--
-- Name: COLUMN reviews.rating_adjusted; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.rating_adjusted IS 'Flag indicating if sentiment was adjusted based on rating';


--
-- Name: COLUMN reviews.identified_themes; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.identified_themes IS 'Comma-separated list of themes identified in the review';


--
-- Name: COLUMN reviews.preprocessed_text; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.preprocessed_text IS 'Cleaned and preprocessed review text for NLP analysis';


--
-- Name: COLUMN reviews.source; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.reviews.source IS 'Data source (Google Play, App Store, etc.)';


--
-- Name: reviews_review_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.reviews_review_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.reviews_review_id_seq OWNER TO postgres;

--
-- Name: reviews_review_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.reviews_review_id_seq OWNED BY public.reviews.review_id;


--
-- Name: banks bank_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks ALTER COLUMN bank_id SET DEFAULT nextval('public.banks_bank_id_seq'::regclass);


--
-- Name: reviews review_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews ALTER COLUMN review_id SET DEFAULT nextval('public.reviews_review_id_seq'::regclass);


--
-- Name: banks banks_app_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks
    ADD CONSTRAINT banks_app_id_key UNIQUE (app_id);


--
-- Name: banks banks_bank_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks
    ADD CONSTRAINT banks_bank_name_key UNIQUE (bank_name);


--
-- Name: banks banks_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks
    ADD CONSTRAINT banks_pkey PRIMARY KEY (bank_id);


--
-- Name: reviews reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT reviews_pkey PRIMARY KEY (review_id);


--
-- Name: idx_reviews_bank_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_bank_id ON public.reviews USING btree (bank_id);


--
-- Name: idx_reviews_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_date ON public.reviews USING btree (review_date);


--
-- Name: idx_reviews_rating; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_rating ON public.reviews USING btree (rating);


--
-- Name: idx_reviews_sentiment; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_sentiment ON public.reviews USING btree (sentiment_label);


--
-- Name: idx_reviews_text_search; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_text_search ON public.reviews USING gin (to_tsvector('english'::regconfig, review_text));


--
-- Name: reviews reviews_bank_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT reviews_bank_id_fkey FOREIGN KEY (bank_id) REFERENCES public.banks(bank_id) ON DELETE CASCADE;


--
-- Name: TABLE banks; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.banks TO analyst;


--
-- Name: SEQUENCE banks_bank_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.banks_bank_id_seq TO analyst;


--
-- Name: TABLE reviews; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.reviews TO analyst;


--
-- Name: SEQUENCE reviews_review_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.reviews_review_id_seq TO analyst;


--
-- PostgreSQL database dump complete
--

\unrestrict 6jyLz9nP6dFLLZZN83StpVCAkAzRq94OhoCJ2ws0iUAhevEzY6VlA5Tu4t88ZIp


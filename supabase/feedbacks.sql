create extension if not exists pgcrypto;

create table if not exists public.feedbacks (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default timezone('utc', now()),
    is_correct boolean not null,
    transcription_text text not null,
    detected_verse jsonb,
    correction jsonb,
    comment text
);

create index if not exists feedbacks_created_at_idx
    on public.feedbacks (created_at desc);

alter table public.feedbacks enable row level security;

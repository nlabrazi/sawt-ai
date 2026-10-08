-- Apply in the Supabase SQL Editor before using the tafsir store.
begin;

create table if not exists public.tafsir_entries (
    surah_id integer not null check (surah_id between 1 and 114),
    ayah integer not null check (ayah between 1 and 286),
    source text not null check (source in ('ibn_kathir', 'as_saadi')),
    text_fr text not null check (text_fr ~ '[^[:space:]]'),
    source_reference text not null check (source_reference ~ '[^[:space:]]'),
    version text not null check (version ~ '[^[:space:]]'),
    status text not null default 'need_review' check (status in ('need_review', 'verified')),
    reviewed_at timestamptz,
    provenance jsonb not null check (
        jsonb_typeof(provenance) = 'object'
        and provenance ?& array[
            'source_language', 'source_edition', 'reuse_reference', 'imported_at',
            'source_text', 'source_surah_id', 'source_start_ayah', 'source_end_ayah'
        ]
    ),
    updated_at timestamptz not null default clock_timestamp(),
    primary key (surah_id, ayah, source),
    constraint tafsir_review_consistency check (
        (status = 'need_review' and reviewed_at is null)
        or (status = 'verified' and reviewed_at is not null)
    )
);

create index if not exists tafsir_entries_review_idx
    on public.tafsir_entries (status, surah_id, source, ayah);

create or replace function public.guard_tafsir_entry_write()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
    if tg_op = 'INSERT' then
        if new.status <> 'need_review' or new.reviewed_at is not null then
            raise exception 'A tafsir import must start with need_review and no review date.'
                using errcode = '23514';
        end if;
    else
        if row(new.surah_id, new.ayah, new.source, new.source_reference, new.version, new.provenance)
           is distinct from
           row(old.surah_id, old.ayah, old.source, old.source_reference, old.version, old.provenance) then
            raise exception 'The original tafsir reference and provenance are immutable.'
                using errcode = '23514';
        end if;

        if new.text_fr is distinct from old.text_fr then
            new.status := 'need_review';
            new.reviewed_at := null;
        elsif new.status = 'verified' and old.status = 'need_review' then
            new.reviewed_at := clock_timestamp();
        elsif new.status = 'need_review' then
            new.reviewed_at := null;
        else
            new.reviewed_at := old.reviewed_at;
        end if;
    end if;

    new.updated_at := clock_timestamp();
    return new;
end;
$$;

drop trigger if exists tafsir_entry_write_guard on public.tafsir_entries;
create trigger tafsir_entry_write_guard
    before insert or update on public.tafsir_entries
    for each row execute function public.guard_tafsir_entry_write();

alter table public.tafsir_entries enable row level security;
revoke all on table public.tafsir_entries from public, anon, authenticated, service_role;
grant select, insert on table public.tafsir_entries to service_role;
grant update (text_fr, status) on table public.tafsir_entries to service_role;
revoke all on function public.guard_tafsir_entry_write() from public, anon, authenticated, service_role;

commit;

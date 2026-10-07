-- Run after tafsir_entries.sql in an empty disposable test database.
-- All text is explicitly fictional. This transaction leaves no test data behind.
\set ON_ERROR_STOP on
begin;
set local role service_role;

do $$
declare
    original jsonb := jsonb_build_object(
        'source_language', 'ar', 'source_edition', 'Édition fictive de test',
        'reuse_reference', 'Conditions fictives de test', 'imported_at', clock_timestamp(),
        'source_text', 'Passage fictif sans contenu religieux.', 'source_surah_id', 2,
        'source_start_ayah', 255, 'source_end_ayah', 255
    );
    initial_revision timestamptz;
    first_review timestamptz;
    current_revision timestamptz;
    affected integer;
begin
    insert into public.tafsir_entries (surah_id, ayah, source, text_fr, source_reference, version, provenance)
    values
        (2, 255, 'ibn_kathir', 'Brouillon Ibn Kathir fictif.', 'Référence fictive 2:255', 'test-1', original),
        (2, 255, 'as_saadi', 'Brouillon As-Saadi fictif.', 'Référence fictive 2:255', 'test-1', original),
        (1, 1, 'ibn_kathir', 'Autre brouillon fictif.', 'Référence fictive 1:1', 'test-1',
            original || jsonb_build_object('source_surah_id', 1, 'source_start_ayah', 1, 'source_end_ayah', 1));

    assert (select count(*) = 3 from public.tafsir_entries where status = 'need_review' and reviewed_at is null),
        'Every inserted row must start pending.';

    begin
        insert into public.tafsir_entries (surah_id, ayah, source, text_fr, source_reference, version, provenance, status, reviewed_at)
        values (2, 1, 'ibn_kathir', 'Faux texte validé de test.', 'Référence fictive', 'test-1', original, 'verified', clock_timestamp());
        raise exception 'A verified import was accepted.';
    exception when check_violation then null;
    end;

    begin
        insert into public.tafsir_entries (surah_id, ayah, source, text_fr, source_reference, version, provenance, reviewed_at)
        values (2, 1, 'ibn_kathir', 'Faux texte de test.', 'Référence fictive', 'test-1', original, clock_timestamp());
        raise exception 'An import with a review date was accepted.';
    exception when check_violation then null;
    end;

    select updated_at into initial_revision from public.tafsir_entries
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir';
    update public.tafsir_entries set status = 'verified'
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir'
        and status = 'need_review' and updated_at = initial_revision;
    get diagnostics affected = row_count;
    assert affected = 1, 'The current review must validate exactly one source and verse.';
    select reviewed_at into first_review from public.tafsir_entries
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir';
    assert first_review is not null, 'Validation must assign a review date.';
    assert (select status = 'need_review' from public.tafsir_entries
        where surah_id = 2 and ayah = 255 and source = 'as_saadi'), 'The other source must remain pending.';
    assert (select count(*) = 1 from public.tafsir_entries
        where surah_id = 2 and ayah between 254 and 255 and status = 'verified'),
        'A public read must expose only the verified verse and source.';

    update public.tafsir_entries set text_fr = 'Correction fictive de test.', status = 'verified'
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir';
    assert (select status = 'need_review' and reviewed_at is null and provenance = original
        from public.tafsir_entries where surah_id = 2 and ayah = 255 and source = 'ibn_kathir'),
        'An edit must cancel verification and preserve the original source.';

    update public.tafsir_entries set status = 'verified'
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir' and updated_at = initial_revision;
    get diagnostics affected = row_count;
    assert affected = 0, 'A stale review must not validate modified text.';

    select updated_at into current_revision from public.tafsir_entries
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir';
    update public.tafsir_entries set status = 'verified'
        where surah_id = 2 and ayah = 255 and source = 'ibn_kathir' and updated_at = current_revision;
    assert (select status = 'verified' and reviewed_at >= first_review and text_fr = 'Correction fictive de test.'
        from public.tafsir_entries where surah_id = 2 and ayah = 255 and source = 'ibn_kathir'),
        'The correction must become visible only after a new validation.';

    begin
        insert into public.tafsir_entries (surah_id, ayah, source, text_fr, source_reference, version, provenance)
        values
            (2, 2, 'ibn_kathir', 'Nouveau brouillon fictif.', 'Référence fictive', 'test-1', original),
            (2, 255, 'ibn_kathir', 'Réimport fictif.', 'Référence fictive', 'test-2', original);
        raise exception 'A duplicate import was accepted.';
    exception when unique_violation then null;
    end;
    assert not exists(select 1 from public.tafsir_entries where surah_id = 2 and ayah = 2),
        'A bulk import conflict must roll back every new entry.';
    assert (select text_fr = 'Correction fictive de test.' and status = 'verified'
        from public.tafsir_entries where surah_id = 2 and ayah = 255 and source = 'ibn_kathir'),
        'A reimport must preserve reviewed text.';

    begin
        update public.tafsir_entries set provenance = '{}'::jsonb;
        raise exception 'The backend role could overwrite source provenance.';
    exception when insufficient_privilege then null;
    end;
    begin
        update public.tafsir_entries set text_fr = E' \t\n '
            where surah_id = 2 and ayah = 255 and source = 'ibn_kathir';
        raise exception 'Whitespace-only text was accepted.';
    exception when check_violation then null;
    end;
end;
$$;

reset role;
do $$
begin
    begin
        update public.tafsir_entries set source_reference = 'Autre référence fictive';
        raise exception 'The write guard allowed a provenance change.';
    exception when check_violation then null;
    end;
end;
$$;

do $$
declare
    client_role text;
begin
    foreach client_role in array array['anon', 'authenticated'] loop
        execute format('set local role %I', client_role);
        begin
            perform text_fr from public.tafsir_entries;
            raise exception '% could read the private table.', client_role;
        exception when insufficient_privilege then null;
        end;
        begin
            insert into public.tafsir_entries (surah_id, ayah, source, text_fr, source_reference, version, provenance)
            values (2, 3, 'ibn_kathir', 'Test fictif.', 'Test', 'test-1', '{}'::jsonb);
            raise exception '% could insert into the private table.', client_role;
        exception when insufficient_privilege then null;
        end;
        begin
            update public.tafsir_entries set status = 'verified';
            raise exception '% could validate a tafsir.', client_role;
        exception when insufficient_privilege then null;
        end;
        begin
            delete from public.tafsir_entries;
            raise exception '% could delete a tafsir.', client_role;
        exception when insufficient_privilege then null;
        end;
        reset role;
    end loop;
end;
$$;

-- Check RLS independently from grants: accidental client SELECT still sees no rows.
grant select on public.tafsir_entries to anon;
set local role anon;
do $$
begin
    assert (select count(*) = 0 from public.tafsir_entries), 'RLS exposed private tafsirs.';
end;
$$;
reset role;
rollback;
\echo 'Tafsir SQL checks passed: import, review, corrections, conflicts, grants and RLS.'

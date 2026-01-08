drop table projects;
drop table pages;
drop table pages_fts;
drop table chunks;
drop table chunks_fts;

create table if not exists projects (
    id integer primary key autoincrement,
    name text not null,
    constraint unique_project_name unique (name)
);

create table if not exists pages (
    id integer primary key autoincrement,
    project_id integer not null,
    name text not null,
    views integer not null,
    status integer null,
    html text null,
    markdown text null,
    constraint unique_project_page unique (project_id, name),
    constraint fk_project foreign key (project_id) references projects(id)
);

create unique index if not exists pages_project_id_name on pages(project_id, name);
create index if not exists pages_project_id on pages(project_id);
create index if not exists pages_views on pages(views);

create virtual table if not exists pages_fts using fts5(
    name, 
    content = 'pages', 
    content_rowid = 'id'
);

create trigger if not exists pages_ai
after insert on pages
begin
    insert into pages_fts(rowid, name)
    values (new.id, new.name);
end;

create trigger if not exists pages_ad
after delete on pages
begin
    insert into pages_fts(pages_fts, rowid)
    values ('delete', old.id);
end;

create trigger if not exists pages_au
after update of id, name on pages
begin
    insert into pages_fts(pages_fts, rowid)
    values ('delete', old.id);
    insert into pages_fts(rowid, name)
    values (new.id, new.name);
end;

create table if not exists chunks (
    id integer primary key autoincrement,
    page_id INTEGER,
    text TEXT,
    status integer null,
    embedding blob,
    constraint unique_page_text unique (page_id, text),
    constraint fk_page foreign key (page_id) references pages(id)
);

create unique index if not exists chunks_page_id_text on chunks(page_id, text);
create index if not exists chunks_page_id on chunks(page_id);

create virtual table if not exists chunks_fts using fts5(
    text, 
    content = 'chunks', 
    content_rowid = 'id'
);

create trigger if not exists chunks_ai
after insert on chunks
begin
    insert into chunks_fts(rowid, text)
    values (new.id, new.text);
end;

create trigger if not exists chunks_ad
after delete on chunks
begin
    insert into chunks_fts(chunks_fts, rowid)
    values ('delete', old.id);
end;

create trigger if not exists chunks_au
after update of id, text on chunks
begin
    insert into chunks_fts(chunks_fts, rowid)
    values ('delete', old.id);
    insert into chunks_fts(rowid, text)
    values (new.id, new.text);
end;


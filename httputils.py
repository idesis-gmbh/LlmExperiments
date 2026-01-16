from html.parser import HTMLParser
from urllib.error import HTTPError
import urllib.request


class WikipediaHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.in_style = False
        self.in_table = False
        self.in_categories = False
        self.in_footer = False
        self.in_headline = False
        self.got_headline = False
        self.in_paragraph = False
        self.got_paragraph = False
        self.list_type = None
        self.in_list = False
        self.markdown = []
        self.toc = []
        self.headlines = []
        self.sections = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "style":
            self.tags.append((tag, dict(attrs)))
            self.in_style = True
        elif tag == "table":
            self.tags.append((tag, dict(attrs)))
            self.in_table = True
        elif tag == "div" and dict(attrs).get("class") == "catlinks":
            self.tags.append((tag, dict(attrs)))
            self.in_categories = True
        elif tag == "footer":
            self.tags.append((tag, dict(attrs)))
            self.in_footer = True
        elif tag.startswith("h") and len(tag) == 2 and tag[1:].isdigit():
            self.tags.append((tag, dict(attrs)))
            if self.got_headline or int(tag[1:]) == 1:
                self.in_headline = True
                self.markdown.append("#" * int(tag[1:]) + " ")
                self.toc.append("#" * int(tag[1:]) + " ")
                self.headlines = self.headlines[: int(tag[1:]) - 1] + [
                    "#" * int(tag[1:]) + " "
                ]
                # self.sections.append(([], []))
        elif (
            self.got_headline
            and not self.in_table
            and not self.in_categories
            and not self.in_footer
            and tag == "p"
        ):
            if not self.tags or self.tags[-1][0] != "p":
                self.tags.append((tag, dict(attrs)))
            self.in_paragraph = True
            self.markdown.append("")
            # self.sections[-1][1].append("")
            self.sections.append((self.headlines[:], [""]))
        elif (
            self.got_headline
            and self.got_paragraph
            and not self.in_table
            and not self.in_categories
            and not self.in_footer
            and tag in ["ul", "ol"]
        ):
            self.tags.append((tag, dict(attrs)))
            if not self.list_type:
                self.sections.append((self.headlines[:], []))
            self.list_type = tag
        elif self.list_type and tag == "li":
            if not self.tags or self.tags[-1][0] != "li":
                self.tags.append((tag, dict(attrs)))
            self.in_list = True
            self.markdown.append("* " if self.list_type == "ul" else "1. ")
            if self.headlines[-1].endswith("Notes"):
                self.sections.append(
                    (self.headlines[:], ["* " if self.list_type == "ul" else "1. "])
                )
            else:
                self.sections[-1][1].append("* " if self.list_type == "ul" else "1. ")
        else:
            self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "style":
            assert self.tags[-1][0] == "style"
            self.tags.pop()
            self.in_style = False
        elif tag == "table":
            # while self.tags and self.tags[-1][0] != tag:
            #     self.tags.pop()
            assert self.tags[-1][0] == "table"
            self.tags.pop()
            self.in_table = any(tag[0] == "table" for tag in self.tags)
        elif tag == "div":
            while self.tags and self.tags[-1][0] != tag:
                self.tags.pop()
            div = self.tags.pop()
            if div[1].get("class") == "catlinks":
                self.in_categories = False
        elif tag == "footer":
            # while self.tags and self.tags[-1][0] != tag:
            #     self.tags.pop()
            assert self.tags[-1][0] == tag
            self.tags.pop()
            self.in_footer = False
        elif tag.startswith("h") and len(tag) == 2 and tag[1:].isdigit():
            # while self.tags and self.tags[-1][0] != tag:
            #     self.tags.pop()
            assert self.tags[-1][0] == tag
            self.tags.pop()
            if self.got_headline or int(tag[1:]) == 1:
                self.markdown[-1] += "\n"
                # self.sections[-1][0][:] = self.headlines[:]
                self.in_headline = False
            if int(tag[1:]) == 1:
                self.got_headline = True
        elif (
            self.got_headline
            and not self.in_table
            and not self.in_categories
            and not self.in_footer
            and tag == "p"
        ):
            while self.tags and self.tags[-1][0] != tag:
                self.tags.pop()
            if self.tags:
                self.tags.pop()
            self.in_paragraph = False
            self.got_paragraph = True
        elif (
            self.got_headline
            and self.got_paragraph
            and not self.in_table
            and not self.in_categories
            and not self.in_footer
            and tag in ["ul", "ol"]
        ):
            # while self.tags and self.tags[-1][0] != tag:
            #     self.tags.pop()
            assert self.tags[-1][0] == tag
            self.tags.pop()
            in_list = [tag[0] for tag in self.tags if tag in ["ul", "ol"]]
            self.in_list = bool(in_list)
            self.list_type = in_list[-1] if in_list else None
        elif self.list_type and tag == "li":
            while self.tags and self.tags[-1][0] != tag:
                self.tags.pop()
            if self.tags:
                self.tags.pop()
            self.markdown[-1] += "\n"
            self.in_list = False
        else:
            while self.tags and self.tags[-1][0] != tag:
                assert self.tags[-1][0] not in ["style", "table"]
                self.tags.pop()
            if self.tags:
                self.tags.pop()

    def handle_data(self, data):
        if (
            not self.in_style
            and not self.in_table
            and not self.in_categories
            and not self.in_footer
        ):
            if self.in_headline or self.in_paragraph or self.in_list:
                self.markdown[-1] += data
            if self.in_headline:
                self.toc[-1] += data
                self.headlines[-1] += data
            elif self.in_paragraph or self.in_list:
                self.sections[-1][1][-1] += data


def get_wikipedia_page(project_name, page_name):
    req = urllib.request.Request(
        url=f"https://{project_name}.org/wiki/{page_name}".encode("utf-8").decode(
            "ascii", "ignore"
        ),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    print("get_wikipedia_page", req.full_url)
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            html = response.read().decode("utf-8")
    except HTTPError as e:
        status = e.status
        html = None
    return status, html

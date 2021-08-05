from IPython.display import HTML, display
from IPython.core.magic import register_cell_magic

def set_styles():
    styles = """<style>div.green { 
                    background-color: #59b571;
                    border-color: #1c8538; 
                    border-left: 5px solid #1c8538; 
                    padding: 0.5em;}
                    div.red { 
                    background-color: #fcf2f2;
                    border-color: #dFb5b4; 
                    border-left: 5px solid #dfb5b4; 
                    padding: 0.5em;}
                    div.yellow { 
                    background-color: #f7eb99;
                    border-color: #edd11a; 
                    border-left: 5px solid #edd11a; 
                    padding: 0.5em;}
                    div.blue { 
                    background-color: #c9deff;
                    border-color: #9cc2ff; 
                    border-left: 5px solid #9cc2ff; 
                    padding: 0.5em;}<style>
                    """

    return  HTML(styles)

def set_background(color):    
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}">'.format(script)))

@register_cell_magic
def background(color, cell):
    set_background(color)




define(["base/js/namespace","base/js/dialog"], function(Jupyter, dialog) {
    function load_ipython_extension() {
        if (!Jupyter.toolbar) return;
        Jupyter.toolbar.add_buttons_group([{
            'label': 'Run object tracking cells',
            'icon': 'fa-play-circle',
            'callback': function() {
                var cells = Jupyter.notebook.get_cells();
                var found = false;
                for (var i = 0; i < cells.length; i++) {
                    var cell = cells[i];
                    if (cell.metadata && cell.metadata.tags && cell.metadata.tags.indexOf('object_tracking') !== -1) {
                        found = true;
                        Jupyter.notebook.execute_cell(i);
                    }
                }
                if (!found) {
                    dialog.modal({
                        title: 'Object Tracking',
                        body: 'No cells with tag "object_tracking" found. Running all code cells instead.',
                        buttons: { 'OK': {} }
                    });
                    Jupyter.notebook.execute_all_cells();
                }
            }
        }]);
    }
    return { load_ipython_extension: load_ipython_extension };
});

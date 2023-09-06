function dadlogo(varargin)

global dad

val = varargin{1};

switch val
    case 1
        % Introduction
        screen = get(0, 'ScreenSize');
        swidth  = screen(3);
        sheight = screen(4);
        
        im = imread('DADlogo.jpg');
        iwidth  = size(im,2) - 1000;
        iheight = size(im,1) - 900;

        pos = [(swidth - iwidth)/2 (sheight-iheight)/2 iwidth iheight];

        dad.gui.fig_logo = figure('Visible', 'on', 'menubar', 'none', 'paperpositionmode', 'auto', 'numbertitle', 'off', ...
            'resize', 'off', 'position', pos, 'name', [dad.intern.name]);

        image(im);
        set(gca, 'Visible', 'off', 'Position', [0 0 1 1]);

        text(600, 50, [dad.intern.versiontxt, '  (', dad.intern.version_datestr, ')'], 'units', 'pixel', ...
            'horizontalalignment', 'left', 'Fontsize', 14, 'Color', [0.1 0.1 0.1]);
        
    case 2
        % Copyright information
        screen = get(0, 'ScreenSize');
        swidth  = screen(3);
        sheight = screen(4);
        
        im = imread('DADlogo.jpg');
        iwidth  = size(im,2) - 1200;
        iheight = size(im,1) - 900;

        pos = [(swidth - iwidth)/2 (sheight-iheight)/2 iwidth iheight];

        dad.gui.fig_logo = figure('Visible', 'on', 'menubar', 'none', 'paperpositionmode', 'auto', 'numbertitle', 'off', ...
            'resize', 'off', 'position', pos, 'name', [dad.intern.name]);

        image(im);
        set(gca, 'Visible', 'off', 'Position', [0 0 1 1]);

        text(400, 120, [dad.intern.versiontxt, '  (', dad.intern.version_datestr, ')'], 'units', 'pixel', ...
            'horizontalalignment', 'left', 'Fontsize', 14, 'Color', [0.1 0.1 0.1]);
        
        text(400, 100, ['Copyright ',  '      ', '2014.'], 'units', 'pixel', ...
            'horizontalalignment', 'left', 'Fontsize', 14, 'Color', [0.1 0.1 0.1]);
        
        text(464, 100, '$\textcopyright$', 'Interpreter', 'latex', 'units', 'pixel', ...
            'horizontalalignment', 'left', 'Fontsize', 14, 'Color', [0.1 0.1 0.1]);
        
        text(400, 80, 'All rights reserved.', 'units', 'pixel', ...
            'horizontalalignment', 'left', 'Fontsize', 14, 'Color', [0.1 0.1 0.1]);
        
end
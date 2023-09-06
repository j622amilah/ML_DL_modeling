function var1 = guilogo(main_path, var1)

screen = get(0,'screensize');
swidth  = screen(3);
sheight = screen(4);

logo_path = [main_path, '\images\guilogo.jpg'];
im = imread(logo_path);

iwidth  = size(im,2);
iheight = size(im,1);

pos = [(swidth-iwidth)/2 (sheight-iheight)/2 iwidth iheight];

var1.fig_logo = figure('visible','on','menubar','none','paperpositionmode','auto','numbertitle','off','resize','off','position',pos,'name',['About ',var1.name]);

image(im);
set(gca,'visible','off','Position',[0 0 1 1]);

% text(30,90, [leda2.intern.versiontxt,'  (',leda2.intern.version_datestr,')'],'units','pixel','horizontalalignment','left','fontsize',14,'color',[.1 .1 .1]);
% text(30,70, 'Code by Mathias Benedek & Christian Kaernbach','units','pixel','horizontalalignment','left','fontsize',8,'color',[.1 .1 .1]);

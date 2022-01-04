import sys
sys.path.insert(0, '/var/www/html/')
sys.path.insert(0, '/var/www/html/Sperm_Morphology_App')
from Sperm_Morphology_App import app as application


<VirtualHost *:80>
		ServerName backend.sperm-morphology.shafieelab.org
		ServerAdmin admin@backend.sperm-morphology.shafieelab.org
		WSGIScriptAlias / /var/www/html/Sperm_Morphology_App/flaskapp.wsgi
        WSGIApplicationGroup %{GLOBAL}

		<Directory /var/www/html/Sperm_Morphology_App/>
			Order allow,deny
			Allow from all
		</Directory>
		ErrorLog /var/www/html/error.log
		LogLevel warn
		CustomLog /var/www/html/access.log  combined
</VirtualHost>


<VirtualHost *:80>
	# The ServerName directive sets the request scheme, hostname and port that
	# the server uses to identify itself. This is used when creating
	# redirection URLs. In the context of virtual hosts, the ServerName
	# specifies what hostname must appear in the request's Host: header to
	# match this virtual host. For the default virtual host (this file) this
	# value is not decisive as it is used as a last resort host regardless.
	# However, you must set it for any further virtual host explicitly.
	#ServerName www.example.com

	ServerAdmin webmaster@localhost
	DocumentRoot /var/www/html
        WSGIDaemonProcess Sperm_Morphology_App  threads=8
        WSGIScriptAlias / /var/www/html/Sperm_Morphology_App/flaskapp.wsgi
        WSGIApplicationGroup %{GLOBAL}
        <Directory flaskapp>
             WSGIProcessGroup flaskapp
             WSGIApplicationGroup %{GLOBAL}
             Order deny,allow
             Allow from all
        </Directory>

	# Available loglevels: trace8, ..., trace1, debug, info, notice, warn,
	# error, crit, alert, emerg.
	# It is also possible to configure the loglevel for particular
	# modules, e.g.
	#LogLevel info ssl:warn

	ErrorLog /var/www/html/error.log
	CustomLog /var/www/html/access.log combined

	# For most configuration files from conf-available/, which are
	# enabled or disabled at a global level, it is possible to
	# include a line for only one particular virtual host. For example the
	# following line enables the CGI configuration for this host only
	# after it has been globally disabled with "a2disconf".
	#Include conf-available/serve-cgi-bin.conf
</VirtualHost>

# vim: syntax=apache ts=4 sw=4 sts=4 sr noet